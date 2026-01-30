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
#include "litert/compiler/mlir/dialects/litert/lazy_resource_blob.h"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "tsl/platform/fingerprint.h"

namespace litert {

LazyResourceBlob::LazyResourceBlob(absl::string_view path,
                                   llvm::sys::fs::file_t fd, size_t alignment,
                                   size_t size, uint64_t hash)
    : done_(false),
      path_(path),
      fd_(fd),
      alignment_(alignment),
      size_(size),
      hash_(hash) {}

LazyResourceBlob::~LazyResourceBlob() {
  CHECK(done_) << "Must call Cleanup before destroying LazyResourceBlob.";
}

LazyResourceBlob::LazyResourceBlob(LazyResourceBlob&& other) {
  done_ = other.done_;
  other.done_ = true;
  alignment_ = other.alignment_;
  path_ = other.path_;
  fd_ = other.fd_;
  size_ = other.size_;
  hash_ = other.hash_;
}

LazyResourceBlob& LazyResourceBlob::operator=(LazyResourceBlob&& other) {
  done_ = other.done_;
  other.done_ = true;
  path_ = other.path_;
  fd_ = other.fd_;
  alignment_ = other.alignment_;
  size_ = other.size_;
  hash_ = other.hash_;

  return *this;
}

LazyResourceBlob LazyResourceBlob::CreateAndCopyData(
    llvm::ArrayRef<uint8_t> data, size_t alignment) {
  llvm::SmallString<32> path;
  int rw_fd = -1;

  std::error_code ec_create_temp_file =
      llvm::sys::fs::createTemporaryFile("lazy-resource-blob", "", rw_fd, path);
  if (ec_create_temp_file) {
    // Crash OK
    LOG(FATAL) << "Couldn't construct temp file to persist resource blob: "
               << ec_create_temp_file;
  }

  absl::string_view data_view(reinterpret_cast<const char*>(data.data()),
                              data.size());
  size_t data_size = data_view.size();
  uint64_t hash = tsl::Fingerprint64(data_view);
  {
    llvm::raw_fd_ostream blob_stream(rw_fd, /*shouldClose=*/true);
    blob_stream.write(data_view.data(), data_view.size());
    if (blob_stream.has_error()) {
      // Crash OK
      LOG(FATAL) << "Couldn't construct temp file to persist resource blob: "
                 << blob_stream.error();
    }
  }

  // After writing the file, we open it again and keep the FD alive until
  // Cleanup is called. This allows the OS to delete the temp file upon program
  // exit.
  llvm::Expected<llvm::sys::fs::file_t> fd_or =
      llvm::sys::fs::openNativeFileForReadWrite(
          path, llvm::sys::fs::CD_OpenExisting, llvm::sys::fs::OF_None);
  if (llvm::Error err = fd_or.takeError()) {
    // Crash OK
    LOG(FATAL) << "Couldn't open temp file: " << llvm::toString(std::move(err));
  }

  // Since we have an open FD, this just marks the file for removal with the OS.
  // Removal will happen when we close the FD. Note that we can no longer
  // conduct operations on the file via the path alone since the path is invalid
  // (i.e. no file exists at the given path). All operations must be conducted
  // on the open FD.
  std::error_code ec_remove =
      llvm::sys::fs::remove(path, /*IgnoreNonExisting=*/false);
  if (ec_remove) {
    // Crash OK
    LOG(FATAL) << "Failed to call `remove` on temp file: " << ec_remove;
  }

  LazyResourceBlob b(static_cast<std::string>(path), *fd_or, alignment,
                     data_size, hash);
  return b;
}

ScopedDataHandle LazyResourceBlob::GetDataHandle() const {
  auto buffer_or = llvm::MemoryBuffer::getOpenFile(
      fd_, path_, /*FileSize=*/-1, /*RequiresNullTerminator=*/false,
      /*IsVolatile=*/false, alignment_);

  if (std::error_code ec = buffer_or.getError()) {
    // Crash OK
    LOG(FATAL) << "Failed to open file at " << path_ << ", error_code: " << ec;
  }

  auto buffer = std::move(buffer_or.get());

  CHECK(llvm::isAddrAligned(alignment_, buffer->getBufferStart()));

  return ScopedDataHandle(std::move(buffer));
}

BlobChunkReader LazyResourceBlob::GetChunkReader(size_t chunk_size) const {
  if (done_) {
    return BlobChunkReader(fd_, 0);
  }
  return BlobChunkReader(fd_, chunk_size);
}

void LazyResourceBlob::Cleanup() {
  if (done_) {
    return;
  }

  std::error_code ec = llvm::sys::fs::closeFile(fd_);
  if (ec) {
    // Crash OK
    LOG(FATAL) << "Failed to close FD to temp file: " << ec;
  }

  done_ = true;
}

}  // namespace litert
