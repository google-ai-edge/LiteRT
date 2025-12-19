// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_LITERT_BUFFER_REF_H_
#define ODML_LITERT_LITERT_CC_LITERT_BUFFER_REF_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert {

//===----------------------------------------------------------------------===//
//
//                                                            << BUFFER REF >>
//
/// @file
/// @brief Provides read-only, read/write, and owning views of buffers.
///
/// Serialized model artifacts and assets are frequently large and non-standard.
/// These classes simplify handling such buffers efficiently without copies.
/// They also provide read and write capabilities with left-padding awareness
/// through standard signed char string types. This is useful for manual edits
/// to FlatBuffer metadata or directly to the serialized FlatBuffer.
///
/// Pre-allocated buffers can be transferred to these classes, or allocation can
/// be handled internally. Buffer references can be implicitly upcasted to
/// non-owning read/write or read-only views for other routines.
///
/// Example:
/// @code
/// void ReadBuffer(BufferRef r_buf) { std::cerr << r_buf.StrView(); }
/// void WriteToBuffer(MutableBufferRef rw_buf) { rw_buf.WriteTo("SomeData"); }
/// ...
/// OwningBuffer<uint8_t> buf(size);
/// WriteToBuffer(buf); // Implicitly convert to a non-owning, read/write view.
/// ReadBuffer(buf); // Implicitly convert to a read-only view.
/// @endcode
//
//===----------------------------------------------------------------------===//

/// @brief Defines allocation and deallocation behavior for owning buffer refs.
///
/// An allocator is a trivially constructible/destructible object that
/// overrides `operator()` for allocating and freeing memory.

/// @brief A memory allocator based on `malloc` and `free`.
/// @tparam ByteT The byte type of the buffer.
template <typename ByteT = uint8_t>
struct Mallocator {
  /// @brief Frees the given memory block.
  void operator()(ByteT* d) {
    if (d != nullptr) {
      free(d);
    }
  }

  /// @brief Allocates a memory block of the specified size.
  /// @param bytes The size of the memory block in bytes.
  /// @return A pointer to the allocated memory.
  ByteT* operator()(size_t bytes) {
    return reinterpret_cast<ByteT*>(malloc(bytes));
  }
};

/// @brief A memory allocator based on `new` and `delete`.
/// @tparam ByteT The byte type of the buffer.
template <typename ByteT = uint8_t>
struct Newlocator {
  /// @brief Deletes the given array.
  void operator()(ByteT* d) {
    if (d != nullptr) {
      delete[] d;
    }
  }

  /// @brief Allocates an array of the specified size.
  /// @param bytes The size of the array in bytes.
  /// @return A pointer to the allocated array.
  ByteT* operator()(size_t bytes) { return new ByteT[bytes]; }
};

//
// Read-Only Bytes
//

/// @brief An immutable and non-owning view of a buffer.
/// @tparam ByteT The byte type of the buffer.
template <typename ByteT = uint8_t>
class BufferRef {
 public:
  using TupleT = std::tuple<const ByteT* const, const size_t, const size_t>;

  /// @brief Constructs a null buffer.
  BufferRef() : end_offset_(0), start_offset_(0), data_(nullptr) {}

  /// @brief Constructs from an already allocated buffer.
  ///
  /// The view will only expose `data[offset, offset + size]`.
  /// @param data A pointer to the start of the buffer.
  /// @param end_offset The end offset of the buffer view.
  /// @param start_offset The start offset of the buffer view.
  BufferRef(const ByteT* data, size_t end_offset, size_t start_offset = 0)
      : end_offset_(end_offset),
        start_offset_(start_offset),
        data_(const_cast<ByteT*>(data)) {}
  BufferRef(const void* data, size_t end_offset, size_t start_offset = 0)
      : end_offset_(end_offset),
        start_offset_(start_offset),
        data_(const_cast<ByteT*>(reinterpret_cast<const ByteT*>(data))) {}
  explicit BufferRef(absl::Span<const ByteT> data)
      : end_offset_(data.size()),
        start_offset_(0),
        data_(const_cast<ByteT*>(data.data())) {}

  /// @brief Returns a pointer to the start of the actual data.
  const ByteT* Data() const { return data_ + start_offset_; }

  /// @brief Returns the size of the actual data.
  size_t Size() const { return end_offset_ - start_offset_; }

  /// @brief Returns the buffer details as a tuple.
  TupleT Get() const { return TupleT(data_, end_offset_, start_offset_); }

  /// @brief Returns a pointer to the start of the actual data as a signed char.
  /// @note The returned pointer might not be null-terminated.
  const char* StrData() const { return reinterpret_cast<const char*>(Data()); }

  /// @brief Returns a string view of the actual data.
  /// @note Ensures the view is null-terminated.
  absl::string_view StrView() const {
    return absl::string_view(StrData(), Size());
  }

  /// @brief Returns a const span of the actual data.
  absl::Span<const ByteT> Span() const {
    return absl::MakeConstSpan(Data(), Size());
  }

  /// @brief Copies the buffer data to a vector.
  std::vector<ByteT> ToVec() const {
    return std::vector<ByteT>(StrData(), StrData() + Size());
  }

  /// @brief Writes the string data to a stream.
  void WriteStr(std::ostream& out) const {
    out.write(StrData(), Size());
    out.flush();
  }

  /// @brief Prints information about this buffer.
  void Dump(std::ostream& out) const {
    out << absl::StreamFormat("%s[%lu:%lu]\n", TypeName(), start_offset_,
                              end_offset_);
  }

  BufferRef(const BufferRef& other) = default;
  BufferRef& operator=(const BufferRef& other) = default;

  virtual ~BufferRef() = default;

 protected:
  /// The end offset of the actual data, relative to `data_`.
  size_t end_offset_;
  /// The start offset of the actual data, relative to `data_`.
  size_t start_offset_;
  /// The original pointer to the memory.
  ByteT* data_ = nullptr;

  /// @brief Returns the debug name of the class.
  virtual absl::string_view TypeName() const { return "BufferRef"; }
};
template <typename ByteT = uint8_t>
BufferRef(const ByteT*, size_t, size_t) -> BufferRef<ByteT>;

//
// Read-Write Non-Owning Bytes
//

/// @brief A writable, non-owning version of `BufferRef`.
/// @tparam ByteT The byte type of the buffer.
template <typename ByteT>
class MutableBufferRef : public BufferRef<ByteT> {
 public:
  using TupleT = std::tuple<ByteT* const, const size_t, const size_t>;

  /// @brief Constructs a null buffer.
  MutableBufferRef()
      : BufferRef<ByteT>((ByteT*)nullptr, /*size*/ 0, /*offset*/ 0) {}

  /// @brief Creates a mutable view from a pre-allocated non-const buffer.
  MutableBufferRef(ByteT* data, size_t size, size_t offset = 0)
      : BufferRef<ByteT>(data, size, offset) {}
  MutableBufferRef(void* data, size_t size, size_t offset = 0)
      : BufferRef<ByteT>(data, size, offset) {}
  explicit MutableBufferRef(absl::Span<ByteT> data) : BufferRef<ByteT>(data) {}
  explicit MutableBufferRef(absl::Span<const ByteT> data) = delete;
  MutableBufferRef(const ByteT*, size_t, size_t) = delete;
  MutableBufferRef(const void*, size_t, size_t) = delete;

  /// @brief Returns a mutable pointer to the start of the actual data.
  ByteT* Data() { return this->data_ + this->start_offset_; }

  /// @brief Returns a mutable char pointer to the start of the actual data.
  char* StrData() { return reinterpret_cast<char*>(Data()); }

  /// @brief Returns the buffer info as a tuple.
  TupleT Get() {
    return TupleT(this->data_, this->end_offset_, this->start_offset_);
  }

  /// @brief Returns a mutable span of the actual data.
  absl::Span<ByteT> Span() { return absl::MakeSpan(Data(), this->Size()); }

  /// @brief Writes a string into the buffer at a specified offset.
  /// @param str The string to write.
  /// @param offset The offset at which to start writing.
  /// @return `true` if the entire string fits and is written, `false`
  /// otherwise.
  bool WriteInto(absl::string_view str, size_t offset = 0) {
    if (str.size() > this->Size() - offset) {
      return false;
    }
    std::memcpy(Data() + offset, str.data(), str.size());
    return true;
  }

  MutableBufferRef(const MutableBufferRef& other) = default;
  MutableBufferRef& operator=(const MutableBufferRef& other) = default;

 protected:
  /// @brief Returns the debug name of the class.
  absl::string_view TypeName() const override { return "MutableBufferRef"; }
};
template <typename ByteT>
MutableBufferRef(ByteT*, size_t, size_t) -> MutableBufferRef<ByteT>;

//
// Read-Write Owning Bytes
//

/// @brief A writable and owning buffer reference.
///
/// This class can allocate new buffers internally and take ownership of
/// existing ones. It does not support resizing.
/// @tparam ByteT The byte type of the buffer.
/// @tparam Allocator The allocator to use for memory management.
template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
class OwningBufferRef : public MutableBufferRef<ByteT> {
 public:
  using TupleT = std::tuple<ByteT* const, const size_t, const size_t>;
  using WeakTupleT = std::tuple<ByteT*&, size_t&, size_t&>;

  /// @brief Constructs a null buffer.
  OwningBufferRef()
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, /*size*/ 0,
                                /*offset*/ 0) {}

  /// @brief Initializes a new buffer reference and allocates it internally.
  /// @param size The size of the buffer to allocate.
  explicit OwningBufferRef(size_t size)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, size, /*offset*/ 0) {
    this->data_ = (ByteT*)Allocator()(size);
  }

  /// @brief Takes ownership of a given buffer.
  OwningBufferRef(ByteT* data, size_t size, size_t offset = 0)
      : MutableBufferRef<ByteT>(data, size, offset) {}
  OwningBufferRef(void* data, size_t size, size_t offset = 0)
      : MutableBufferRef<ByteT>(data, size, offset) {}
  explicit OwningBufferRef(absl::Span<ByteT> data)
      : MutableBufferRef<ByteT>(data) {}

  /// @brief Copies the given buffer.
  OwningBufferRef(const ByteT* data, size_t size)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, size,
                                /*offset*/ 0) {
    this->data_ = (ByteT*)Allocator()(size);
    std::memcpy(this->data_, data, size);
  }
  explicit OwningBufferRef(absl::Span<const ByteT> data)
      : OwningBufferRef<ByteT, Allocator>(data.data(), data.size()) {}

  /// @brief Copies data from a given string.
  explicit OwningBufferRef(absl::string_view data)
      : OwningBufferRef<ByteT, Allocator>(
            reinterpret_cast<const ByteT*>(data.data()), data.size()) {}

  /// @brief Copies data from a given C-style string.
  explicit OwningBufferRef(const char* data)
      : OwningBufferRef<ByteT, Allocator>(absl::string_view(data)) {}

  /// @brief Drops the reference to any owned memory.
  void Drop() {
    this->data_ = nullptr;
    this->end_offset_ = 0;
    this->start_offset_ = 0;
  }

  /// @brief Returns the buffer details and drops references to them.
  TupleT Release() {
    auto res =
        std::make_tuple(this->data_, this->end_offset_, this->start_offset_);
    Drop();
    return res;
  }

  /// @brief Returns weak references to buffer data.
  ///
  /// Takes ownership of any data that is swapped in.
  WeakTupleT GetWeak() {
    return WeakTupleT(this->data_, this->end_offset_, this->start_offset_);
  }

  /// @brief Frees any owned memory.
  void Reset() {
    Allocator()(this->data_);
    Drop();
  }

  /// @brief Resets any existing data and copies in the given buffer.
  void Assign(const ByteT* buf, size_t end_offset, size_t start_offset = 0) {
    Reset();
    this->end_offset_ = end_offset;
    this->data_ = (ByteT*)Allocator()(this->end_offset_);
    std::memcpy(this->data_, buf, this->end_offset_);
    this->start_offset_ = start_offset;
  }

  OwningBufferRef(OwningBufferRef&& other)
      : MutableBufferRef<ByteT>(other.data_, other.end_offset_,
                                other.start_offset_) {
    other.Drop();
  }

  OwningBufferRef& operator=(OwningBufferRef&& other) {
    if (this != &other) {
      Reset();
      this->data_ = other.data_;
      this->end_offset_ = other.end_offset_;
      this->start_offset_ = other.start_offset_;
      other.Drop();
    }
    return *this;
  }

  OwningBufferRef(const OwningBufferRef& other)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, other.end_offset_,
                                other.start_offset_) {
    Assign(other.data_, other.end_offset_, other.start_offset_);
  }

  OwningBufferRef& operator=(const OwningBufferRef& other) {
    Assign(other.data_, other.end_offset_, other.start_offset_);
    return *this;
  }

  ~OwningBufferRef() override { Reset(); }

 protected:
  /// @brief Returns the debug name of the class.
  absl::string_view TypeName() const override { return "OwningBufferRef"; }
};

template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
OwningBufferRef(const ByteT*, size_t) -> OwningBufferRef<ByteT, Allocator>;

template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
OwningBufferRef(ByteT*, size_t) -> OwningBufferRef<ByteT, Allocator>;

template <typename ByteT = char, class Allocator = Newlocator<ByteT>>
OwningBufferRef(const char*) -> OwningBufferRef<ByteT, Allocator>;

template <typename Iter>
OwningBufferRef<uint8_t> MakeBufferRef(Iter begin, Iter end) {
  using T = typename std::remove_reference_t<
      std::remove_cv_t<typename std::iterator_traits<Iter>::value_type>>;
  const size_t element_size = sizeof(*begin);
  const size_t num_elements = std::distance(begin, end);
  OwningBufferRef res(num_elements * element_size);
  std::copy(begin, end, reinterpret_cast<T*>(res.Data()));
  return res;
}

template <typename T>
OwningBufferRef<uint8_t> MakeBufferRef(std::initializer_list<T> data) {
  return MakeBufferRef(std::cbegin(data), std::cend(data));
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_BUFFER_REF_H_
