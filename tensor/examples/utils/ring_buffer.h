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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_RING_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_RING_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/utils/macros.h"

namespace litert::tensor::examples {

// Memory of a `RingBuffer`, with positions and sizes in bytes.
//
// Use `RingBuffer<T>` instead, which works with elements.
class RingBufferStorage {
 public:
  enum class Mode {
    // `kMirrored` when the platform supports it, `kCompacting` otherwise.
    kAuto,
    // The same memory is mapped twice back to back: the live bytes are
    // always contiguous without ever being moved.
    kMirrored,
    // A buffer of twice the capacity. The live bytes are moved to its start
    // when they would run past its end, i.e. at most once every `capacity`
    // appended bytes.
    kCompacting,
  };

  // Number of bytes after the end of the live bytes that are always readable.
  //
  // Backends like XNNPACK may read a few bytes past the end of their inputs.
  // These bytes hold unspecified values.
  static constexpr size_t kReadablePaddingBytes = 64;

  // Returns true if `Mode::kMirrored` is available: the platform supports it
  // and the process is allowed to make the syscalls that it needs, which
  // sandboxes like seccomp may forbid. Checked once, on the first call.
  static bool IsMirroredModeSupported();

  // Creates a storage for up to `capacity_bytes > 0` bytes.
  static absl::StatusOr<RingBufferStorage> Create(size_t capacity_bytes,
                                                  Mode mode = Mode::kAuto);

  RingBufferStorage(RingBufferStorage&& other) noexcept;
  RingBufferStorage& operator=(RingBufferStorage&& other) noexcept;
  ~RingBufferStorage();

  // `kMirrored` or `kCompacting`, never `kAuto`.
  Mode mode() const { return mode_; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  // Returns the first live byte.
  std::byte* data();
  const std::byte* data() const;

  // The methods below leave the storage unchanged when they return an error.

  // Appends `bytes` bytes with unspecified values, dropping the oldest bytes to
  // make room, and returns a pointer to the first new byte.
  //
  // Returns an `InvalidArgumentError` if `bytes > capacity()`.
  absl::StatusOr<std::byte*> ExtendBack(size_t bytes);

  // Sets the size to `bytes`, adding bytes with unspecified values at the back
  // or dropping bytes from the back.
  //
  // Returns an `InvalidArgumentError` if `bytes > capacity()`.
  absl::Status Resize(size_t bytes);

  // Drops the `bytes` oldest bytes.
  //
  // Returns an `OutOfRangeError` if `bytes > size()`.
  absl::Status PopFront(size_t bytes);

  // Drops every byte.
  void Clear();

 private:
  RingBufferStorage(Mode mode, std::byte* base, size_t capacity,
                    size_t buffer_size, size_t allocation_size)
      : mode_(mode),
        base_(base),
        capacity_(capacity),
        buffer_size_(buffer_size),
        allocation_size_(allocation_size) {}

  // Unmaps or frees the memory.
  void Release();

  // Drops the `bytes <= size_` oldest bytes.
  void DropFront(size_t bytes);

  Mode mode_ = Mode::kCompacting;
  // - kMirrored: start of a mapping of `buffer_size_` bytes of memory, mapped
  //   again right after it. One more page of it follows as padding if
  //   `buffer_size_ - capacity_ < kReadablePaddingBytes`.
  // - kCompacting: start of a heap allocation.
  std::byte* base_ = nullptr;
  size_t capacity_ = 0;
  // - kMirrored: size of the memory that is mapped twice (`capacity_` rounded
  //   up to the page size).
  // - kCompacting: `2 * capacity_`, the bytes that the live
  //   bytes can occupy before they are moved back to the start.
  size_t buffer_size_ = 0;
  // Size of the whole mapping or allocation that starts at `base_`.
  size_t allocation_size_ = 0;
  // Offset of the first live byte from `base_`. Below `buffer_size_` in the
  // mirrored mode.
  size_t start_ = 0;
  size_t size_ = 0;
};

// Tag for `RingBuffer::resize` to leave the new elements
// uninitialized.
struct NoInit {
  explicit NoInit() = default;
};
inline constexpr NoInit kNoInit{};

// A fixed capacity ring buffer whose elements are always contiguous in memory.
//
// Pushing to a full buffer drops the front elements.
//
// `T` must be trivially copyable and its size must be a power of two.
template <class T>
class RingBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "RingBuffer elements must be trivially copyable.");
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                "RingBuffer element size must be a power of two.");

 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using Mode = RingBufferStorage::Mode;

  // Number of bytes after the last element that are always readable. See
  // `RingBufferStorage::kReadablePaddingBytes`.
  static constexpr size_t kReadablePaddingBytes =
      RingBufferStorage::kReadablePaddingBytes;

  // Creates a buffer holding up to `capacity > 0` elements.
  static absl::StatusOr<RingBuffer> Create(size_t capacity,
                                           Mode mode = Mode::kAuto) {
    if (capacity == 0) {
      return absl::InvalidArgumentError(
          "RingBuffer capacity must be positive.");
    }
    if (capacity > std::numeric_limits<size_t>::max() / 4 / sizeof(T)) {
      return absl::InvalidArgumentError("RingBuffer capacity is too large.");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        RingBufferStorage storage,
        RingBufferStorage::Create(capacity * sizeof(T), mode));
    return RingBuffer(std::move(storage));
  }

  RingBuffer(RingBuffer&&) noexcept = default;
  RingBuffer& operator=(RingBuffer&&) noexcept = default;

  // `kMirrored` or `kCompacting`, never `kAuto`.
  Mode mode() const { return storage_.mode(); }

  size_type size() const { return storage_.size() / sizeof(T); }
  size_type capacity() const { return storage_.capacity() / sizeof(T); }
  bool empty() const { return storage_.size() == 0; }

  // The pointers and iterators are invalidated by the next call to a method
  // that changes the size.
  T* data() { return reinterpret_cast<T*>(storage_.data()); }
  const T* data() const { return reinterpret_cast<const T*>(storage_.data()); }

  iterator begin() { return data(); }
  const_iterator begin() const { return data(); }
  const_iterator cbegin() const { return data(); }
  iterator end() { return data() + size(); }
  const_iterator end() const { return data() + size(); }
  const_iterator cend() const { return data() + size(); }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const { return rbegin(); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const { return rend(); }

  // `i < size()`.
  T& operator[](size_type i) { return data()[i]; }
  const T& operator[](size_type i) const { return data()[i]; }

  // The buffer must not be empty.
  T& front() { return data()[0]; }
  const T& front() const { return data()[0]; }
  T& back() { return data()[size() - 1]; }
  const T& back() const { return data()[size() - 1]; }

  // Appends `value`, dropping the front element if the buffer is full.
  void push_back(const T& value) {
    // Can't fail: `capacity() >= 1`.
    absl::StatusOr<std::byte*> new_bytes = storage_.ExtendBack(sizeof(T));
    std::memcpy(*new_bytes, &value, sizeof(T));
  }

  // Appends `values`, dropping front elements to make room. Only the last
  // `capacity()` elements of `values` are kept if it holds more.
  void append_range(absl::Span<const T> values) {
    if (values.size() > capacity()) {
      values.remove_prefix(values.size() - capacity());
    }
    if (values.empty()) {
      return;
    }
    // Can't fail: `values.size() <= capacity()`.
    absl::StatusOr<std::byte*> new_bytes =
        storage_.ExtendBack(values.size() * sizeof(T));
    std::memcpy(*new_bytes, values.data(), values.size() * sizeof(T));
  }

  // The methods below leave the buffer unchanged when they return an error.

  // Drops the front element.
  //
  // Returns an `OutOfRangeError` if the buffer is empty.
  absl::Status pop_front() { return pop_front(1); }

  // Drops the `count` front elements.
  //
  // Returns an `OutOfRangeError` if `count > size()`.
  absl::Status pop_front(size_type count) {
    if (count > size()) {
      return absl::OutOfRangeError(
          absl::StrCat("Can't pop ", count, " elements from a RingBuffer of ",
                       size(), " elements."));
    }
    return storage_.PopFront(count * sizeof(T));
  }

  // Sets the size to `count`, never dropping front elements: growing appends
  // value-initialized elements, shrinking drops elements from the back.
  //
  // Returns an `InvalidArgumentError` if `count > capacity()`.
  absl::Status resize(size_type count) { return resize(count, T()); }

  // Same as above, but growing appends copies of `value`.
  absl::Status resize(size_type count, const T& value) {
    // `value` may be an element that growing moves.
    const T copy = value;
    const size_type old_size = size();
    LRT_TENSOR_RETURN_IF_ERROR(resize(count, kNoInit));
    if (count > old_size) {
      std::fill(begin() + old_size, end(), copy);
    }
    return absl::OkStatus();
  }

  // Same as above, but growing appends elements with unspecified values for
  // the caller to overwrite.
  absl::Status resize(size_type count, NoInit) {
    if (count > capacity()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Can't resize a RingBuffer of capacity ", capacity(),
                       " to ", count, " elements."));
    }
    return storage_.Resize(count * sizeof(T));
  }

  // Drops every element.
  void clear() { storage_.Clear(); }

 private:
  explicit RingBuffer(RingBufferStorage storage)
      : storage_(std::move(storage)) {}

  RingBufferStorage storage_;
};

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_RING_BUFFER_H_
