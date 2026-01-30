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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_RESOURCE_BLOB_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_RESOURCE_BLOB_H_

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"

namespace litert {

class BlobChunkReader {
 public:
  class Iterator {
   public:
    // Iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = absl::string_view;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;

   private:
    explicit Iterator(BlobChunkReader* reader)
        : reader_(reader), buffer_(reader->chunk_size_) {
      if (reader->chunk_size_ <= 0) {
        is_end_ = true;
        return;
      }

      int fd;
      // Since we've already unlinked the original file we need to create a new
      // FD from /proc/self/fd.
      std::error_code ec = llvm::sys::fs::openFileForRead(
          absl::StrCat("/proc/self/fd/", reader_->fd_), fd);
      if (ec) {
        LOG(FATAL) << "Failed to open underlying data file: " << ec.message();
      }
      stream_ = std::make_unique<llvm::raw_fd_stream>(fd,
                                                      /*shouldClose=*/true);
      ReadNextChunk();  // Perform initial read
    }

    // Private constructor for end() - default construction marks as end
    Iterator() : is_end_(true) {}

    void ReadNextChunk() {
      if (!stream_ || is_end_) {
        is_end_ = true;
        return;
      }

      size_t bytes_read = stream_->read(buffer_.data(), reader_->chunk_size_);

      if (bytes_read == -1 || stream_->has_error()) {
        LOG(FATAL) << "Error reading file chunk: "
                   << stream_->error().message();
      }

      buffer_.resize(bytes_read);
      if (bytes_read == 0) {  // EOF condition
        is_end_ = true;
        stream_ = nullptr;
        buffer_.clear();
      }
    }

    // Allow BlobChunkReader to call the private constructors
    friend class BlobChunkReader;

    // Pointer back to the reader instance to access filename and chunk_size
    BlobChunkReader* reader_ = nullptr;
    std::unique_ptr<llvm::raw_fd_stream> stream_ = nullptr;
    std::vector<char> buffer_;
    bool is_end_ = false;

   public:
    // Default constructor creates an end iterator (publicly accessible if
    // needed, but mainly for end()) Iterator() : is_end_(true) {} // Defined
    // above

    // Dereference operator
    reference operator*() const {
      return absl::string_view(buffer_.data(), buffer_.size());
    }

    // Prefix increment operator
    Iterator& operator++() {
      if (!is_end_) {
        ReadNextChunk();
      }
      return *this;
    }

    // Comparison operator
    bool operator!=(const Iterator& other) const {
      // Comparison is primarily based on the end state.
      // For input iterators, comparing non-end iterators isn't always
      // meaningful, but comparing against end() is the key.
      return is_end_ != other.is_end_;
    }

    bool operator==(const Iterator& other) const {
      return is_end_ == other.is_end_;
    }

    // Make iterator non-copyable, but movable
    Iterator(const Iterator&) = delete;
    Iterator& operator=(const Iterator&) = delete;
    Iterator(Iterator&&) = default;
    Iterator& operator=(Iterator&&) = default;
  };

  explicit BlobChunkReader(llvm::sys::fs::file_t fd, size_t chunk_size)
      : fd_(fd), chunk_size_(chunk_size) {}

  Iterator begin() { return Iterator(this); }
  Iterator end() { return Iterator(); }

  // Make the range object movable
  BlobChunkReader(BlobChunkReader&&) = default;
  BlobChunkReader& operator=(BlobChunkReader&&) = default;

  // Keep non-copyable
  BlobChunkReader(const BlobChunkReader&) = delete;
  BlobChunkReader& operator=(const BlobChunkReader&) = delete;

 private:
  llvm::sys::fs::file_t fd_;
  size_t chunk_size_;
};

class [[nodiscard]] ScopedDataHandle {
 public:
  explicit ScopedDataHandle(std::unique_ptr<llvm::MemoryBuffer> buffer)
      : buffer_(std::move(buffer)) {}

  llvm::ArrayRef<uint8_t> GetRawData() const {
    return llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t*>(buffer_->getBufferStart()),
        buffer_->getBufferSize());
  }

  template <typename T>
  llvm::ArrayRef<T> GetDataAs() const {
    auto raw_data = GetRawData();
    return llvm::ArrayRef<T>(reinterpret_cast<const T*>(raw_data.data()),
                             raw_data.size() / sizeof(T));
  }

 private:
  std::unique_ptr<llvm::MemoryBuffer> buffer_;
};

class LazyResourceBlob {
 public:
  LazyResourceBlob() = delete;
  LazyResourceBlob(const LazyResourceBlob&) = delete;
  LazyResourceBlob& operator=(const LazyResourceBlob&) = delete;
  LazyResourceBlob(LazyResourceBlob&& other);
  LazyResourceBlob& operator=(LazyResourceBlob&& other);
  ~LazyResourceBlob();

  static LazyResourceBlob CreateAndCopyData(llvm::ArrayRef<uint8_t> data,
                                            size_t alignment);

  template <typename T>
  static LazyResourceBlob CreateAndCopyData(llvm::ArrayRef<T> data) {
    return CreateAndCopyData({reinterpret_cast<const uint8_t*>(data.data()),
                              data.size() * sizeof(T)},
                             alignof(T));
  }

  ScopedDataHandle GetDataHandle() const;
  BlobChunkReader GetChunkReader(size_t chunk_size) const;

  void Cleanup();

  size_t size() const { return size_; }
  uint64_t hash() const { return hash_; }

 private:
  LazyResourceBlob(absl::string_view path, llvm::sys::fs::file_t fd,
                   size_t alignment, size_t size, uint64_t hash);

  bool done_ = false;
  std::string path_;
  llvm::sys::fs::file_t fd_;
  llvm::Align alignment_;
  size_t size_ = 0;
  uint64_t hash_ = 0;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_LAZY_RESOURCE_BLOB_H_
