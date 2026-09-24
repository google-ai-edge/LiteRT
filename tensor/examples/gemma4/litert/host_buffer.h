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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_HOST_BUFFER_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_HOST_BUFFER_H_
#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
namespace litert::tensor::examples::gemma4::cpu {
// Move-only aligned storage. Resize preserves capacity, but not contents.
// 64 initialized padding bytes follow the requested logical span.
class HostBuffer {
 public:
  HostBuffer() = default;
  HostBuffer(HostBuffer&&) = default;
  HostBuffer& operator=(HostBuffer&&) = default;
  void Resize(size_t bytes) {
    if (bytes + 64 > capacity_) {
      capacity_ = bytes + 64;
      data_.reset(static_cast<std::byte*>(
          ::operator new(capacity_, std::align_val_t(64))));
      std::fill_n(data_.get(), capacity_, std::byte{0});
    }
    size_ = bytes;
  }
  std::byte* data() { return data_.get(); }
  const std::byte* data() const { return data_.get(); }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

 private:
  struct Delete {
    void operator()(std::byte* p) const {
      ::operator delete(p, std::align_val_t(64));
    }
  };
  std::unique_ptr<std::byte, Delete> data_;
  size_t size_ = 0, capacity_ = 0;
};
}  // namespace litert::tensor::examples::gemma4::cpu
#endif
