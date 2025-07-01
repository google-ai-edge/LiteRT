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
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_rng.h"
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
    return LiteRtAlignedMem(reinterpret_cast<T*>(std::aligned_alloc(
        LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, num_elements * sizeof(T))));
  }

 public:
  // Create a buffer with the given tensor type.
  static Expected<SimpleBuffer> Create(RankedTensorType tensor_type) {
    LITERT_ASSIGN_OR_RETURN(const size_t bytes, tensor_type.Bytes());
    auto data_ptr = MakeAlloc(bytes);
    LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
    return SimpleBuffer(std::move(data_ptr), std::move(env),
                        std::move(tensor_type), bytes);
  }

  // Create buffer with the given shape and type.
  template <typename T>
  static Expected<SimpleBuffer> Create(
      std::initializer_list<Layout::Dim> dimensions) {
    return Create(MakeRankedTensorType<T>(std::move(dimensions)));
  }

  // Create a new buffer with the provided type information and data.
  template <typename T>
  static Expected<SimpleBuffer> Create(
      std::initializer_list<Layout::Dim> dimensions,
      std::initializer_list<T> data) {
    LITERT_ASSIGN_OR_RETURN(auto helper, Create<T>(std::move(dimensions)));
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

  // Create a buffer with same size and type information as the provided
  // tensor buffer, and fill it with random data. Data generation is dictated
  // by the traits template.
  template <typename Traits, typename Rng>
  static Expected<SimpleBuffer> RandomLikeTensorBuffer(
      Rng& rng, const TensorBuffer& tensor_buffer);

  // Returns a span of const values from the buffer.
  template <typename T = uint8_t>
  absl::Span<const T> ConstSpan() {
    return absl::MakeConstSpan(reinterpret_cast<const T*>(buffer_.get()),
                               TypedNumElements<T>());
  }

  // Returns a span of values from the buffer.
  template <typename T = uint8_t>
  absl::Span<T> Span() {
    return absl::MakeSpan(reinterpret_cast<T*>(buffer_.get()),
                          TypedNumElements<T>());
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

  // Writes random values into the contained buffer based on the provided
  // traits.
  template <typename Traits, typename Rng>
  Expected<void> WriteRandom(Rng& rng, size_t start = 0,
                             std::optional<size_t> num_elements = {});

  // Create a new native tensor buffer from this buffer which points to the
  // underlying host memory.
  Expected<TensorBuffer> SpawnTensorBuffer() {
    return TensorBuffer::CreateFromHostMemory(tensor_type_, buffer_.get(),
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

 private:
  SimpleBuffer(LiteRtAlignedMem buffer, Environment env,
               RankedTensorType tensor_type, size_t size_in_bytes)
      : buffer_(std::move(buffer)),
        env_(std::move(env)),
        tensor_type_(std::move(tensor_type)),
        size_in_bytes_(size_in_bytes) {}

  LiteRtAlignedMem buffer_;
  Environment env_;
  RankedTensorType tensor_type_;
  size_t size_in_bytes_;
};

// Populate the given tensor buffer like object with random data. Note that
// the DataGen param carries the cc primitive type.
template <typename DataGen, typename TensorBuf, typename Rng>
Expected<void> PopulateRandomTensorBufferImpl(Rng& rng, TensorBuf& buf,
                                              size_t start,
                                              size_t num_elements) {
  DataGen data_gen;
  LITERT_ASSIGN_OR_RETURN(const auto vec, data_gen(rng, num_elements));
  return buf.Write(absl::MakeConstSpan(vec), start);
}

// Default trait class for configuring rng behavior based on datatype.
// Users can create their own for custom configuration.
struct BasicRandomTensorBufferTraits {
  template <typename D>
  using Gen = SelectT<std::is_floating_point<D>, RandomTensorData<D>,
                      std::is_integral<D>, RandomTensorData<D>>;
};

// Populate the given tensor buffer like object with random data based on the
// provided tensor type. The Trats parameter can be used to configure the
// random number generator per-datatype.
template <typename Traits, typename TensorBuf, typename Rng>
Expected<void> PopulateRandomTensorBuffer(Rng& rng, TensorBuf& buf,
                                          const LiteRtRankedTensorType& ty,
                                          size_t start, size_t num_elements) {
  const auto element_type = ty.element_type;
  if (element_type == kLiteRtElementTypeFloat32) {
    using Gen = typename Traits::template Gen<float>;
    return PopulateRandomTensorBufferImpl<Gen>(rng, buf, start, num_elements);
  } else if (element_type == kLiteRtElementTypeInt32) {
    using Gen = typename Traits::template Gen<int>;
    return PopulateRandomTensorBufferImpl<Gen>(rng, buf, start, num_elements);
  }
  // TODO: Finish for all data types.
  return Error(kLiteRtStatusErrorUnsupported, "Unsupported element type");
}

template <typename Traits, typename Rng>
Expected<SimpleBuffer> SimpleBuffer::RandomLikeTensorBuffer(
    Rng& rng, const TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto buf, LikeTensorBuffer(rng, tensor_buffer));
  LITERT_ASSIGN_OR_RETURN(auto type, tensor_buffer.TensorType());
  const auto ty = static_cast<LiteRtRankedTensorType>(type);
  return PopulateRandomTensorBuffer<Traits>(rng, buf, ty);
}

template <typename Traits, typename Rng>
Expected<void> SimpleBuffer::WriteRandom(Rng& rng, size_t start,
                                         std::optional<size_t> num_elements) {
  const size_t n =
      num_elements ? *num_elements : TypedNumElements<uint8_t>() - start;
  const auto ty = static_cast<LiteRtRankedTensorType>(Type());
  return PopulateRandomTensorBuffer<Traits>(rng, *this, ty, start, n);
}

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_SIMPLE_BUFFER_H_
