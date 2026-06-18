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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"

namespace litert::tensor {

static constexpr size_t kCpuBufferAlignment = 64;

// Provides access to data stored in a buffer.
//
// Unlocks the buffer when destroyed.
template <class T>
class LockedBufferSpan {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  // `const std::byte*` if `T` is `const`, otherwise `std::byte`.
  using MaybeConstByte =
      std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;

  LockedBufferSpan(MaybeConstByte* data,
                   std::function<void(MaybeConstByte*)> unlock, size_t bytes)
      : data_(data, std::move(unlock)), bytes_(bytes) {}

  LockedBufferSpan(
      std::unique_ptr<MaybeConstByte, std::function<void(MaybeConstByte*)>>
          data,
      size_t bytes)
      : data_(std::move(data)), bytes_(bytes) {}

  static LockedBufferSpan Empty() {
    return LockedBufferSpan(nullptr, [](MaybeConstByte*) {}, 0);
  }

  // Casts the span to a specific type.
  //
  // Warning: This transfers the lock management to the returned
  // `LockedBufferSpan`.
  template <class U>
  [[nodiscard]] LockedBufferSpan<U> As() && {
    static_assert(
        std::is_const_v<U> || !std::is_const_v<T>,
        "Cannot cast from a constant buffer span to a non constant one.");
    return LockedBufferSpan<U>(std::move(data_), bytes_);
  }

  T* data() const { return reinterpret_cast<T*>(data_.get()); }
  size_t size() const { return bytes_ / sizeof(T); }
  T* begin() { return data(); }
  T* end() { return data() + size(); }
  const T* begin() const { return data(); }
  const T* end() const { return data() + size(); }
  const T* cbegin() const { return data(); }
  const T* cend() const { return data() + size(); }

 private:
  std::unique_ptr<MaybeConstByte, std::function<void(MaybeConstByte*)>> data_;
  size_t bytes_;
};

// The main interface for buffers.
class Buffer {
 public:
  virtual ~Buffer() = default;

  // Returns the type identifier of the buffer.
  //
  // This should generally be implemented as:
  //
  // ```cpp
  // internal::TypeId GetTypeId() const override {
  //   return internal::TypeId::Get<TYPE_OF_BUFFER_DERIVED_CLASS>();
  // }
  // ```
  virtual internal::TypeId GetTypeId() const = 0;

  // Returns true if the buffer concrete type corresponds to the given type
  // identifier.
  //
  // This should generally be implemented as:
  //
  // ```cpp
  // bool IsA(internal::TypeId id) const override {
  //   return id == internal::TypeId::Get<TYPE_OF_BUFFER_DERIVED_CLASS>() ||
  //      BASE_CLASS::IsA(id);
  // }
  // ```
  virtual bool IsA(internal::TypeId id) const = 0;

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Warning: the RAII object may not manage the data lifetime, only whether
  // it's accessible from the cpu or not.
  virtual LockedBufferSpan<const std::byte> Lock() = 0;

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading and writing the data.
  //
  // On error, returns an empty span.
  virtual LockedBufferSpan<std::byte> LockMutable() {
    return LockedBufferSpan<std::byte>::Empty();
  }

  // Tries to cast the buffer to the given type T.
  template <typename T>
  absl::StatusOr<T&> As() {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<T*>(this);
    }
    return absl::InvalidArgumentError("Wrong buffer type.");
  }

  // Tries to cast the buffer to the given type T.
  template <typename T>
  absl::StatusOr<const T&> As() const {
    if (IsA(internal::TypeId::Get<T>())) {
      return *static_cast<const T*>(this);
    }
    return absl::InvalidArgumentError("Wrong buffer type.");
  }
};

// Provides a view to constant data.
class SpanCpuBuffer : public Buffer {
 public:
  SpanCpuBuffer() = default;
  ~SpanCpuBuffer() override = default;

  // Creates a viewing buffer holding `data` of size `bytes`.
  SpanCpuBuffer(const std::byte* data, size_t bytes)
      : bytes_(bytes), data_(const_cast<std::byte*>(data)) {}

  // Creates a viewing buffer from a C++ array.
  template <class T, size_t N>
  explicit SpanCpuBuffer(const std::array<T, N>& array)
      : SpanCpuBuffer(reinterpret_cast<const std::byte*>(array.data()),
                      sizeof(T) * N) {}

  // Creates a viewing buffer from a C array.
  template <class T, size_t N>
  explicit SpanCpuBuffer(const T (&arr)[N])
      : SpanCpuBuffer(reinterpret_cast<const std::byte*>(arr), sizeof(arr)) {}

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<SpanCpuBuffer>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<SpanCpuBuffer>();
  }

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Note: For CPU buffers, (un)locking is a no-op.
  LockedBufferSpan<const std::byte> Lock() override {
    return {data_, [](const std::byte*) {}, bytes_};
  }

  // Because we know that a CPU buffer is always accessible, we can provide
  // direct accessors to the underlying data.

  const std::byte* data() const { return data_; }
  const std::byte* begin() const { return data_; }
  const std::byte* cbegin() const { return data_; }
  const std::byte* end() const { return data_ + bytes_; }
  const std::byte* cend() const { return data_ + bytes_; }
  size_t size() const { return bytes_; }

  template <class T>
  absl::Span<T> Span() const {
    static_assert(std::is_const_v<T>, "SpanCpuBuffer data is not mutable.");
    return absl::Span<T>(reinterpret_cast<T*>(data_), bytes_ / sizeof(T));
  }

 protected:
  size_t bytes_ = 0;
  std::byte* data_ = nullptr;
};

// Provides a view to mutable data.
class MutableSpanCpuBuffer : public SpanCpuBuffer {
 public:
  MutableSpanCpuBuffer() = default;
  ~MutableSpanCpuBuffer() override = default;

  // Creates a viewing buffer holding `data` of size `bytes`.
  MutableSpanCpuBuffer(std::byte* data, size_t bytes)
      : SpanCpuBuffer(data, bytes) {}

  // Creates a viewing buffer from a C array.
  template <class T, size_t N>
  explicit MutableSpanCpuBuffer(T (&arr)[N]) : SpanCpuBuffer(arr) {}

  // Don't create a mutable span buffer over constant data.
  template <class T, size_t N>
  explicit MutableSpanCpuBuffer(const T (&arr)[N]) = delete;

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<MutableSpanCpuBuffer>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<MutableSpanCpuBuffer>() ||
           SpanCpuBuffer::IsA(id);
  }

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Note: For CPU buffers, (un)locking is a no-op.
  LockedBufferSpan<std::byte> LockMutable() override {
    return {data_, [](std::byte*) {}, bytes_};
  }

  // Because we know that a CPU buffer is always accessible, we can provide
  // direct accessors to the underlying data.

  std::byte* data() const { return data_; }
  std::byte* begin() const { return data_; }
  std::byte* end() const { return data_ + bytes_; }

  template <class T>
  absl::Span<T> Span() const {
    return absl::Span<T>(reinterpret_cast<T*>(data_), bytes_ / sizeof(T));
  }
};

// Manages tensor data.
class OwningCpuBuffer : public Buffer {
 protected:
  static constexpr struct PassKey {
  } kPass{};

 public:
  using CustomAllocPtr = std::unique_ptr<std::byte, void (*)(std::byte*)>;

  ~OwningCpuBuffer() override = default;

  // Builds an owning cpu buffer.
  //
  // Note: This is an internal constructor that is made public to allow building
  // smart pointers in the factory functions that are below by using the passkey
  // idiom.
  OwningCpuBuffer(PassKey, std::shared_ptr<std::byte> data, size_t bytes)
      : data_(std::move(data)), bytes_(bytes) {}

  // We want to avoid unintentional copies.
  OwningCpuBuffer(const OwningCpuBuffer&) = delete;
  OwningCpuBuffer& operator=(const OwningCpuBuffer&) = delete;

  OwningCpuBuffer(OwningCpuBuffer&& other)
      : data_(std::move(other.data_)), bytes_(std::exchange(other.bytes_, 0)) {}

  OwningCpuBuffer& operator=(OwningCpuBuffer&& other) {
    data_ = std::move(other.data_);
    bytes_ = std::exchange(other.bytes_, 0);
    return *this;
  }

  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<OwningCpuBuffer>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<OwningCpuBuffer>();
  }

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  LockedBufferSpan<const std::byte> Lock() override {
    return {data(), [d = data_](const std::byte*) mutable { d.reset(); },
            size()};
  }

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading and writing the data.
  LockedBufferSpan<std::byte> LockMutable() override {
    return {data(), [d = data_](std::byte*) mutable { d.reset(); }, size()};
  }

  template <class T, class = std::enable_if_t<std::is_const_v<T>>>
  absl::Span<T> Span() const {
    return absl::Span<T>(reinterpret_cast<T*>(data()), size() / sizeof(T));
  }

  template <class T>
  absl::Span<T> Span() {
    return absl::Span<T>(reinterpret_cast<T*>(data()), size() / sizeof(T));
  }

  // Because we know that a CPU buffer is always accessible, we can provide
  // direct accessors to the underlying data.

  std::byte* data() { return data_.get(); }
  const std::byte* data() const { return data_.get(); }
  size_t size() const { return bytes_; }
  std::byte* begin() { return data_.get(); }
  std::byte* end() { return data_.get() + bytes_; }
  const std::byte* begin() const { return data_.get(); }
  const std::byte* end() const { return data_.get() + bytes_; }
  const std::byte* cbegin() const { return begin(); }
  const std::byte* cend() const { return end(); }

  // Transfers ownership of `data` to a new `OwningCpuBuffer`.
  static absl::StatusOr<std::shared_ptr<OwningCpuBuffer>> Own(
      CustomAllocPtr data, size_t bytes);

  // Transfers ownership of `data` to a new `OwningCpuBuffer`.
  static absl::StatusOr<std::shared_ptr<OwningCpuBuffer>> Own(
      std::shared_ptr<std::byte> data, size_t bytes);

#define LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, TYPE, ...) \
  case Type::k##TYPE:                                  \
    return OP<Type::k##TYPE>(__VA_ARGS__)

#define LITERT_TENSOR_BUFFER_OP_AS_SWITCH(OP, ...)          \
  switch (type) {                                           \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, BOOL, __VA_ARGS__); \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I2, __VA_ARGS__);   \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I4, __VA_ARGS__);   \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I8, __VA_ARGS__);   \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I16, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I32, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, I64, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, U4, __VA_ARGS__);   \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, U8, __VA_ARGS__);   \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, U16, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, U32, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, U64, __VA_ARGS__);  \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, FP16, __VA_ARGS__); \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, FP32, __VA_ARGS__); \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, FP64, __VA_ARGS__); \
    LITERT_TENSOR_BUFFER_OP_AS_CASE(OP, BF16, __VA_ARGS__); \
    case Type::kUnknown:                                    \
      return nullptr;                                       \
  }

  // Allocates a buffer that can hold `count` elements.
  template <Type type>
  static std::shared_ptr<OwningCpuBuffer> Allocate(size_t count) {
    const size_t bytes = NativeStorage<type>::BufferSize(count);
    CustomAllocPtr copied_data = AlignedAlloc(bytes);
    return std::make_shared<OwningCpuBuffer>(kPass, std::move(copied_data),
                                             bytes);
  }

  template <Type type>
  static std::shared_ptr<OwningCpuBuffer> Allocate(std::vector<int32_t> shape) {
    return Allocate<type>(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  }

  static std::shared_ptr<OwningCpuBuffer> AllocateAs(Type type, size_t count) {
    LITERT_TENSOR_BUFFER_OP_AS_SWITCH(Allocate, count);
  }

  static std::shared_ptr<OwningCpuBuffer> AllocateAs(
      Type type, std::vector<int32_t> shape) {
    LITERT_TENSOR_BUFFER_OP_AS_SWITCH(Allocate, std::move(shape));
  }

  // Builds an `OwningCpuBuffer` and copy the given data to it.
  //
  // Note: This is not done with a constructor to force copies to be explicit.
  static std::shared_ptr<OwningCpuBuffer> Copy(const char* data, size_t bytes);

  // Builds an `OwningCpuBuffer` by copying the elements in the given sequence.
  //
  // - `type`: The underlying storage type.
  template <Type type, class Sequence>
  static std::shared_ptr<OwningCpuBuffer> Copy(Sequence&& seq) {
    using std::begin;
    using std::end;
    using std::size;
    using NativeType = typename NativeStorage<type>::type;
    constexpr Type seq_type =
        ApiType<std::decay_t<decltype(*std::begin(seq))>>::value;
    std::shared_ptr<OwningCpuBuffer> copied_data = Allocate<type>(size(seq));
    if constexpr (type == seq_type) {
      std::copy(begin(seq), end(seq), copied_data->Span<NativeType>().data());
    } else {
      std::transform(begin(seq), end(seq),
                     copied_data->Span<NativeType>().data(),
                     Conversion<type, seq_type>::Call);
    }
    return copied_data;
  }

  // Builds an `OwningCpuBuffer` by copying the elements of the given
  // initializer list.
  template <Type type, class T>
  static std::shared_ptr<OwningCpuBuffer> Copy(std::initializer_list<T>&& seq) {
    return OwningCpuBuffer::Copy<type>(seq);
  }

  // Builds an `OwningCpuBuffer` by copying the elements in the given sequence
  // with run time type dispatching.
  template <class Sequence>
  static std::shared_ptr<OwningCpuBuffer> CopyAs(Type type, Sequence&& seq) {
    LITERT_TENSOR_BUFFER_OP_AS_SWITCH(Copy, std::forward<Sequence>(seq));
  }

  // Builds an `OwningCpuBuffer` by copying the elements of the given
  // initializer list with run time dispatching.
  template <class T>
  static std::shared_ptr<OwningCpuBuffer> CopyAs(
      Type type, std::initializer_list<T>&& seq) {
    LITERT_TENSOR_BUFFER_OP_AS_SWITCH(Copy, std::move(seq));
  }

  // Builds an `OwningCpuBuffer` by applying the given `transform` to elements
  // of a sequence.
  template <Type type, class Sequence, class F>
  static std::shared_ptr<OwningCpuBuffer> Transform(Sequence&& seq,
                                                    F&& transform) {
    using NativeType = typename NativeStorage<type>::type;
    using std::begin;
    using std::end;
    using std::size;
    std::shared_ptr<OwningCpuBuffer> copied_data = Allocate<type>(size(seq));
    std::transform(begin(seq), end(seq), copied_data->Span<NativeType>().data(),
                   std::forward<F>(transform));
    return copied_data;
  }

  // Builds an `OwningCpuBuffer` by applying the given `transform` to elements
  // of a sequence with run time type dispatching.
  template <class Sequence, class F>
  static std::shared_ptr<OwningCpuBuffer> TransformAs(Type type, Sequence&& seq,
                                                      F&& transform) {
    LITERT_TENSOR_BUFFER_OP_AS_SWITCH(Transform, std::forward<Sequence>(seq),
                                      std::forward<F>(transform));
  }

  // Allocates an array with an alignment of `kBufferAlignment`.
  //
  // Note: This is different from `std::aligned_alloc` because it doesn't
  // require the array size to be a multiple of the alignment.
  //
  // To do so we allocate a bigger buffer than requested, get the first
  // aligned address in it and prepend the offset to the real allocation.
  //
  // ```
  // [data    ][off][aligned_data... ]
  //                ↑
  //      This pointer is returned.
  // ```
  static CustomAllocPtr AlignedAlloc(size_t bytes);

  // Frees an array allocated with `AlignedAlloc`.
  static void AlignedFree(std::byte* ptr);

  // Returns true if the given pointer is aligned to `kBufferAlignment`.
  static bool IsAligned(const void* ptr) {
    return !(reinterpret_cast<uintptr_t>(ptr) % kCpuBufferAlignment);
  }

 protected:
  // We use a shared_ptr to be able to lock the buffer and keep the data alive.
  std::shared_ptr<std::byte> data_;
  size_t bytes_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_
