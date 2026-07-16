// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_MMAP_HANDLE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_MMAP_HANDLE_H_

#if defined(_WIN32)
#include <windows.h>
#endif  // defined(_WIN32)

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"

namespace ml_drift {

using ::litert::ml_drift::ReleaseDataCallback;

// Calls the provided callback at then end of the scope this was created into.
template <class F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F&& callback) : callback_(std::forward<F>(callback)) {}
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&& other)
      : active_(other.active_), callback_(std::move(other.callback_)) {
    other.Deactivate();
  }
  ScopeGuard& operator=(ScopeGuard&& other) {
    if (this != &other) {
      active_ = std::move(other.active_);
      callback_ = std::move(other.callback_);
      other.Deactivate();
    }
  }

  ~ScopeGuard() {
    if (active_) {
      callback_();
    }
  }

  void Deactivate() { active_ = false; }

 private:
  F callback_;
  bool active_ = true;
};

template <class F>
ScopeGuard(F&&) -> ScopeGuard<F>;

// Handles MMap allocations lifetime.
//
// When mapped, provides a view over the allocation for convenience.
class MMapHandle {
 public:
  using value_type = uint8_t;
  static constexpr char kUnspecifiedPath[] = "[unspecified]";

  MMapHandle() = default;
  ~MMapHandle();
  MMapHandle(const MMapHandle&) = delete;
  MMapHandle& operator=(const MMapHandle&) = delete;
  MMapHandle(MMapHandle&&);
  MMapHandle& operator=(MMapHandle&&);

  // Maps the file at the given path.
  //
  // If the size is 0, the size of the file minus the offset will be used.
  absl::Status Map(const char* path, size_t offset = 0, size_t size = 0);

  // Maps the fd associated to the file descriptor.
  //
  // If the size is 0, the size of the file minus the offset will be used.
  // The debug_path is printed along the error messages.
  absl::Status Map(const FileDescriptor& fd, size_t offset = 0, size_t size = 0,
                   const char* debug_path = kUnspecifiedPath);

  // Unmaps an existing mapping.
  void UnMap();

  // Returns true if a mapping exists.
  bool IsMapped() const { return data_ != nullptr; }

  // This is used to release the ownership of the mmaped memory to the caller.
  // This function will reset the internal state of the MMapHandle to its
  // default state and return a callback that will release the memory when
  // called.
  ReleaseDataCallback TakeOwnership();

  // Returns the mapping buffer.
  uint8_t* data() { return data_ + offset_page_adjustment_; }

  // Returns the mapping buffer.
  const uint8_t* data() const { return data_ + offset_page_adjustment_; }

  // Returns the mapping size in bytes.
  size_t size() const { return size_; }

  // Returns the offset of the mmaped memory relative to the start of the file.
  size_t offset() const { return offset_; }

  // Returns the offset of the mmaped memory relative to the start of the page.
  //
  // This is useful because mmaping must be done at the start of a page. So if
  // the user requested offset is not at the start of a page, we must start
  // mapping at offset - offset_page_adjustment and we must map a size equal to
  // the user requested size + offset_page_adjustment.
  size_t offset_page_adjustment() const { return offset_page_adjustment_; }

  uint8_t* begin() { return data(); }

  const uint8_t* begin() const { return data(); }

  uint8_t* end() { return data() + size(); }

  const uint8_t* end() const { return data() + size(); }

  friend void swap(MMapHandle& a, MMapHandle& b);

 private:
  // Define an UnMap helper function so that it can be used in both the public
  // UnMap() function and the TakeOwnership() function.
#if defined(_WIN32)
  static void UnMap(uint8_t* data, size_t size, size_t offset_page_adjustment,
                    HANDLE file_mapping);
#else   // defined(_WIN32)
  static void UnMap(uint8_t* data, size_t size, size_t offset_page_adjustment);
#endif  // defined(_WIN32)

  // Resets the internal state of the MMapHandle to its default state.
  // This function does NOT unmap the memory. Use with caution!
  //
  // This is defined as a private separate function so that the public UnMap()
  // function and TakeOwnership() functions can reuse the same reset logic.
  void ResetWithoutUnmapping();

  size_t size_ = 0;
  size_t offset_ = 0;
  size_t offset_page_adjustment_ = 0;
  uint8_t* data_ = nullptr;
#if defined(_WIN32)
  HANDLE file_mapping_ = 0;
#endif  // defined(_WIN32)
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_SERIALIZATION_WEIGHT_CACHE_MMAP_HANDLE_H_
