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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_PY_BLOB_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_PY_BLOB_H_

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/ADT/Hashing.h"

namespace litert {

// A blob of data that is backed by a Python bytes getter function. The getter
// is a python function that returns a generator of bytes.
class PyBlob {
 public:
  using ApplyDataFuncT = absl::FunctionRef<absl::Status(absl::string_view)>;
  PyBlob() = default;

  explicit PyBlob(PyObject* bytes_getter, size_t size)
      : bytes_getter_(bytes_getter), size_(size) {
    if (bytes_getter_) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_XINCREF(bytes_getter_);
      PyGILState_Release(gstate);
    }
  }

  PyBlob(const PyBlob& other)
      : bytes_getter_(other.bytes_getter_), size_(other.size_) {
    if (bytes_getter_) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_XINCREF(bytes_getter_);
      PyGILState_Release(gstate);
    }
  }

  PyBlob& operator=(const PyBlob& other) {
    if (this != &other) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_XDECREF(bytes_getter_);
      bytes_getter_ = other.bytes_getter_;
      size_ = other.size_;
      Py_XINCREF(bytes_getter_);
      PyGILState_Release(gstate);
    }
    return *this;
  }

  PyBlob(PyBlob&& other) noexcept
      : bytes_getter_(std::exchange(other.bytes_getter_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  PyBlob& operator=(PyBlob&& other) noexcept {
    if (this != &other) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_XDECREF(bytes_getter_);
      PyGILState_Release(gstate);
      bytes_getter_ = std::exchange(other.bytes_getter_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~PyBlob() {
    if (bytes_getter_) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_XDECREF(bytes_getter_);
      PyGILState_Release(gstate);
    }
  }

  bool operator==(const PyBlob& other) const {
    return bytes_getter_ == other.bytes_getter_ && size_ == other.size_;
  }

  // Apply the data to the callback.
  // The callback is called with a string_view of the data.
  // The callback must not retain the string_view beyond the call.
  absl::Status ApplyData(ApplyDataFuncT callback) const;

  size_t size() const { return size_; }
  PyObject* bytes_getter() const { return bytes_getter_; }

  // Return the raw address as the hash for MLIR uniqueing
  uint64_t hash() const {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bytes_getter_));
  }

 private:
  PyObject* bytes_getter_ = nullptr;
  size_t size_ = 0;
};

static_assert(!std::is_trivially_copyable<PyBlob>::value);

// LLVM Hashing for MLIR StorageUniquer
inline llvm::hash_code hash_value(const PyBlob& blob) {
  return llvm::hash_value(blob.hash());
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_PY_BLOB_H_
