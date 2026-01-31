// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_SIMPLE_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_SIMPLE_BUFFER_H_

// Misc utility functionality used to implement the cts test suite.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace testing {

// Provides a quality of life wrapper around tensor buffers on host memory
// created in test contexts. This provides management of dependent allocations
// and objects, as well as a type and literal friendly interfaces for
// initializing buffers useful for test contexts.
class SimpleBuffer {
 private:
  struct FreeDeleter {
    void operator()(void* ptr) const { free(ptr); }
  };
  // TODO: Update litert::OwningBufferRef to use support aligned allocs.
  using LiteRtAlignedMem = std::unique_ptr<uint8_t, FreeDeleter>;
  template <typename T = uint8_t>
  static LiteRtAlignedMem MakeAlloc(size_t num_elements) {
    void* host_memory_ptr;
    ABSL_CHECK_EQ(
        posix_memalign(&host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                       num_elements * sizeof(T)),
        0);
    auto res = LiteRtAlignedMem(reinterpret_cast<uint8_t*>(host_memory_ptr));
    return res;
  }

 public:
  using Ref = std::reference_wrapper<SimpleBuffer>;
  using CRef = std::reference_wrapper<const SimpleBuffer>;

  template <typename T>
  struct View {
    absl::Span<T> data;
    absl::Span<const Layout::Dim> dimensions;
    using Type = std::remove_const_t<T>;
    Layout::Dim NumElements() const {
      Layout::Dim num_elements = 1;
      for (Layout::Dim dim : dimensions) {
        num_elements *= dim;
      }
      return num_elements;
    }
  };

  template <typename T>
  using CView = View<const T>;

  // Create a buffer with the given tensor type.
  static Expected<SimpleBuffer> Create(RankedTensorType tensor_type) {
    LITERT_ASSIGN_OR_RETURN(const size_t bytes, tensor_type.Bytes());
    auto data_ptr = MakeAlloc(bytes);
    LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
    return SimpleBuffer(std::move(data_ptr), std::move(env),
                        std::move(tensor_type), bytes);
  }

  // Create buffer with the given shape and type.
  template <typename T, typename Shape>
  static Expected<SimpleBuffer> Create(const Shape& dimensions) {
    return Create(MakeRankedTensorType<T>(dimensions));
  }

  // Create buffer with the given shape and type.
  template <typename T>
  static Expected<SimpleBuffer> Create(
      std::initializer_list<Layout::Dim> dimensions) {
    LITERT_ASSIGN_OR_RETURN(auto helper,
                            Create<T>(Dimensions(std::move(dimensions))));
    return helper;
  }

  // Create a new buffer with the provided type information and literal data.
  template <typename T, typename Shape>
  static Expected<SimpleBuffer> Create(const Shape& dimensions,
                                       std::initializer_list<T> data) {
    LITERT_ASSIGN_OR_RETURN(auto helper, Create<T>(dimensions));
    LITERT_RETURN_IF_ERROR(helper.Write(std::move(data)));
    return helper;
  }

  // Create a new buffer with the provided type information and literal data.
  template <typename T>
  static Expected<SimpleBuffer> Create(
      std::initializer_list<Layout::Dim> dimensions,
      std::initializer_list<T> data) {
    LITERT_ASSIGN_OR_RETURN(auto helper, Create<T>(dimensions));
    LITERT_RETURN_IF_ERROR(helper.Write(std::move(data)));
    return helper;
  }

  // Create a zeroed buffer with the same type information as the
  // provided tensor buffer.
  static Expected<SimpleBuffer> LikeTensorBuffer(
      const TensorBuffer& tensor_buffer) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor_buffer.TensorType());
    return Create(type);
  }

  // Create a new buffer, with the same type information as the provided
  // tensor buffer and a copy of its data.
  static Expected<SimpleBuffer> FromTensorBuffer(
      const TensorBuffer& tensor_buffer) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor_buffer.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto helper, Create(type));
    LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(tensor_buffer)
                               .Read<uint8_t>(helper.Span<uint8_t>()));
    return helper;
  }

  // Create a list of buffers that match the i/o of a graph.
  template <typename It>
  static Expected<std::vector<SimpleBuffer>> LikeSignature(It start, It end) {
    std::vector<SimpleBuffer> buffers;
    while (start != end) {
      Tensor t(*start);
      LITERT_ASSIGN_OR_RETURN(auto type, t.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto b, Create(type));
      buffers.push_back(std::move(b));
      ++start;
    }
    return buffers;
  }

  // Create a buffer with same size and type information as the provided
  // tensor buffer, and fill it with random data. Data generation is dictated
  // by the traits template.
  // TODO: Add visit type pattern to allow skipping explicitly specializing
  // by data type.
  template <typename T, template <typename> typename Generator, typename Rng>
  Expected<void> WriteRandom(Rng& rng, size_t start = 0,
                             std::optional<size_t> num_elements = {}) {
    RandomTensorData<T, Generator> gen;
    return RandomTensorFunctor()(gen, rng, start, num_elements, *this);
  }

  template <typename T, typename Rng>
  Expected<void> WriteRandom(const RandomTensorDataBuilder& b, Rng& rng,
                             size_t start = 0,
                             std::optional<size_t> num_elements = {}) {
    return b.Call<T, RandomTensorFunctor>(rng, start, num_elements, *this);
  }

  template <typename Rng>
  Expected<void> WriteRandom(const RandomTensorDataBuilder& b, Rng& rng,
                             size_t start = 0,
                             std::optional<size_t> num_elements = {}) {
    if (Type().ElementType() == ElementType::Float32) {
      return b.Call<float, RandomTensorFunctor>(rng, start, num_elements,
                                                *this);
    } else if (Type().ElementType() == ElementType::Int32) {
      return b.Call<int32_t, RandomTensorFunctor>(rng, start, num_elements,
                                                  *this);
    } else if (Type().ElementType() == ElementType::Int64) {
      return b.Call<int64_t, RandomTensorFunctor>(rng, start, num_elements,
                                                  *this);
    }
    // TODO: Add support for other types.
    return Error(kLiteRtStatusErrorInvalidArgument, "Unsupported element type");
  }

  // Returns a span of const values from the buffer.
  template <typename T = uint8_t>
  absl::Span<const T> Span() const {
    return absl::MakeConstSpan(reinterpret_cast<const T*>(buffer_.get()),
                               TypedNumElements<T>());
  }

  // Returns a span of values from the buffer.
  template <typename T = uint8_t>
  absl::Span<T> Span() {
    return absl::MakeSpan(reinterpret_cast<T*>(buffer_.get()),
                          TypedNumElements<T>());
  }

  // Return a typed view of both the data and dimensions.
  template <typename T>
  CView<T> AsView() const {
    return {Span<T>(), Type().Layout().Dimensions()};
  }

  // Return a typed view of both the data and dimensions.
  template <typename T>
  View<T> AsView() {
    return {Span<T>(), Type().Layout().Dimensions()};
  }

  // Writes the the provided data into the contained buffer.
  template <typename Arg>
  Expected<void> Write(const Arg& data, size_t start = 0) {
    return Write(std::cbegin(data), std::cend(data), start);
  }

  // Writes the the provided literal data into the contained buffer.
  template <typename T>
  Expected<void> Write(std::initializer_list<T> data, size_t start = 0) {
    return Write(std::cbegin(data), std::cend(data), start);
  }

  // Writes the the provided data into the contained buffer.
  template <typename Iter>
  Expected<void> Write(Iter begin, Iter end, size_t start = 0) {
    using T = typename std::iterator_traits<Iter>::value_type;
    const size_t input_num_elements = std::distance(begin, end);
    const auto elements_to_write =
        std::min(input_num_elements, TypedNumElements<T>() - start);
    std::copy(begin, begin + elements_to_write,
              reinterpret_cast<T*>(buffer_.get()) + start);
    return {};
  }

  // Create a new native tensor buffer from this buffer which points to the
  // underlying host memory.
  Expected<TensorBuffer> SpawnTensorBuffer() const {
    return TensorBuffer::CreateFromHostMemory(env_, tensor_type_, buffer_.get(),
                                              size_in_bytes_);
  }

  // Size in bytes of the underlying buffer.
  size_t Size() const { return size_in_bytes_; }

  // Mutable view of the underlying buffer.
  MutableBufferRef<uint8_t> MutableData() {
    return MutableBufferRef<uint8_t>(buffer_.get(), size_in_bytes_);
  }

  // Immutable view of the underlying buffer.
  BufferRef<uint8_t> Data() const {
    return BufferRef<uint8_t>(buffer_.get(), size_in_bytes_);
  }

  // The tensor type associated with the underlying buffer.
  const RankedTensorType& Type() const { return tensor_type_; }

  // The element type associated with the underlying buffer.
  ElementType ElementType() const { return Type().ElementType(); }

  // Get the size of the underlying buffer up to the largest multiple of the
  // size of the given type.
  template <typename T>
  size_t TypedSize() const {
    return (size_in_bytes_ / sizeof(T)) * sizeof(T);
  }

  // Get the number of elements that can fit in this buffer against the given
  // type.
  template <typename T>
  size_t TypedNumElements() const {
    return TypedSize<T>() / sizeof(T);
  }

  // SimpleBuffer is move-only.
  SimpleBuffer(SimpleBuffer&& other) = default;
  SimpleBuffer& operator=(SimpleBuffer&& other) = default;
  SimpleBuffer(const SimpleBuffer&) = delete;
  SimpleBuffer& operator=(const SimpleBuffer&) = delete;

 private:
  SimpleBuffer(LiteRtAlignedMem buffer, Environment env,
               RankedTensorType tensor_type, size_t size_in_bytes)
      : buffer_(std::move(buffer)),
        env_(std::move(env)),
        tensor_type_(std::move(tensor_type)),
        size_in_bytes_(size_in_bytes) {}

  struct RandomTensorFunctor {
    template <typename Gen, typename Rng>
    Expected<void> operator()(Gen& gen, Rng& rng, size_t start,
                              std::optional<size_t> num_elements,
                              SimpleBuffer& self) {
      const auto num_elements_to_write =
          num_elements
              ? *num_elements
              : self.TypedNumElements<typename Gen::DataType>() - start;
      return gen(rng, self.template Span<typename Gen::DataType>().subspan(
                          start, num_elements_to_write));
    }
  };

  LiteRtAlignedMem buffer_;
  Environment env_;
  RankedTensorType tensor_type_;
  size_t size_in_bytes_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_SIMPLE_BUFFER_H_
