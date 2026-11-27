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

#include "tensor/examples/utils/ring_buffer.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

// Check if SYS_memfd_create is available.
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(SYS_memfd_create)
#define LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED 1
#endif  // defined(SYS_memfd_create)
#endif  // defined(__linux__) || defined(__ANDROID__)

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

namespace litert::tensor::examples::internal {
namespace {

constexpr std::align_val_t kCompactingAlignVal{
    RingBufferStorage::kCompactingAlignment};

#ifdef LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED

size_t RoundUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Closes a file descriptor when it goes out of scope.
class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ~ScopedFd() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  int get() const { return fd_; }

 private:
  int fd_;
};

#endif  // LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED

}  // namespace

bool RingBufferStorage::IsMirroredModeSupported() {
#ifdef LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED
  // The syscalls may be forbidden at run time, e.g. by a seccomp sandbox, so
  // check once that a mirrored storage can be created.
  static const bool supported = [] {
    const absl::StatusOr<RingBufferStorage> probe = Create(1, Mode::kMirrored);
    if (!probe.ok()) {
      ABSL_LOG(INFO) << "Mirrored ring buffers are unavailable: "
                     << probe.status();
    }
    return probe.ok();
  }();
  return supported;
#else
  return false;
#endif
}

absl::StatusOr<RingBufferStorage> RingBufferStorage::Create(
    const size_t capacity_bytes, const Mode mode) {
  if (capacity_bytes == 0) {
    return absl::InvalidArgumentError(
        "RingBufferStorage capacity must be positive.");
  }
  // Keeps the allocation and mapping sizes below from overflowing.
  if (capacity_bytes > std::numeric_limits<size_t>::max() / 4) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RingBufferStorage capacity is too large: ", capacity_bytes,
        " bytes."));
  }
  if (mode == Mode::kAuto) {
    if (!IsMirroredModeSupported()) {
      return Create(capacity_bytes, Mode::kCompacting);
    }
    absl::StatusOr<RingBufferStorage> mirrored =
        Create(capacity_bytes, Mode::kMirrored);
    if (mirrored.ok()) {
      return mirrored;
    }
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Falling back to a compacting ring buffer: " << mirrored.status();
    return Create(capacity_bytes, Mode::kCompacting);
  }

  if (mode == Mode::kCompacting) {
    const size_t allocation_size = 2 * capacity_bytes + kReadablePaddingBytes;
    void* allocation =
        ::operator new(allocation_size, kCompactingAlignVal, std::nothrow);
    if (allocation == nullptr) {
      return absl::ResourceExhaustedError(
          "Could not allocate the ring buffer memory.");
    }
    // The padding is read by backends and must hold initialized values.
    std::memset(allocation, 0, allocation_size);
    return RingBufferStorage(
        Mode::kCompacting, static_cast<std::byte*>(allocation), capacity_bytes,
        2 * capacity_bytes, allocation_size);
  }

#ifdef LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED
  const int64_t page_size_or_error = sysconf(_SC_PAGESIZE);
  if (page_size_or_error <= 0) {
    return absl::ErrnoToStatus(errno, "Could not query the page size");
  }
  const size_t page_size = static_cast<size_t>(page_size_or_error);
  const size_t buffer_size = RoundUp(capacity_bytes, page_size);
  // The live bytes start in the first copy of the memory, so they and their
  // padding end inside the second copy if the capacity leaves room for the
  // padding in the memory. Otherwise, one more page of the memory after the two
  // copies provides the padding.
  static_assert(kReadablePaddingBytes <= 4096);
  const size_t padding_size =
      buffer_size - capacity_bytes >= kReadablePaddingBytes ? 0 : page_size;
  const size_t mapping_size = 2 * buffer_size + padding_size;

  const ScopedFd fd(static_cast<int>(
      syscall(SYS_memfd_create, "litert_tensor_ring_buffer", MFD_CLOEXEC)));
  if (fd.get() < 0) {
    return absl::ErrnoToStatus(errno, "memfd_create failed");
  }
  if (ftruncate(fd.get(), static_cast<off_t>(buffer_size)) != 0) {
    return absl::ErrnoToStatus(errno, "ftruncate failed");
  }

  // Reserve the address range, then map the memory over it.
  void* const reserved = mmap(nullptr, mapping_size, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reserved == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "Could not reserve the address range");
  }
  std::byte* const base = static_cast<std::byte*>(reserved);
  const std::pair<size_t, size_t> views[] = {
      {0, buffer_size},
      {buffer_size, buffer_size},
      {2 * buffer_size, padding_size},
  };
  for (const auto& [offset, size] : views) {
    if (size == 0) {
      continue;
    }
    if (mmap(base + offset, size, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_FIXED, fd.get(), 0) == MAP_FAILED) {
      const int error = errno;
      munmap(base, mapping_size);
      return absl::ErrnoToStatus(error, "Could not map the ring buffer memory");
    }
  }
  return RingBufferStorage(Mode::kMirrored, base, capacity_bytes, buffer_size,
                           mapping_size);
#else
  return absl::UnimplementedError(
      "The mirrored ring buffer mode isn't supported on this platform.");
#endif  // LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED
}

RingBufferStorage::RingBufferStorage(RingBufferStorage&& other) noexcept
    : mode_(other.mode_),
      base_(std::exchange(other.base_, nullptr)),
      capacity_(std::exchange(other.capacity_, 0)),
      buffer_size_(std::exchange(other.buffer_size_, 0)),
      allocation_size_(std::exchange(other.allocation_size_, 0)),
      start_(std::exchange(other.start_, 0)),
      size_(std::exchange(other.size_, 0)) {}

RingBufferStorage& RingBufferStorage::operator=(
    RingBufferStorage&& other) noexcept {
  if (this != &other) {
    Release();
    mode_ = other.mode_;
    base_ = std::exchange(other.base_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    buffer_size_ = std::exchange(other.buffer_size_, 0);
    allocation_size_ = std::exchange(other.allocation_size_, 0);
    start_ = std::exchange(other.start_, 0);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

RingBufferStorage::~RingBufferStorage() { Release(); }

absl::StatusOr<std::byte*> RingBufferStorage::ExtendBack(const size_t bytes) {
  if (bytes > capacity_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Can't extend a RingBufferStorage of capacity ", capacity_,
                     " by ", bytes, " bytes."));
  }
  CompilerBarrier();
  if (size_ + bytes > capacity_) {
    DropFront(size_ + bytes - capacity_);
  }
  if (mode_ == Mode::kCompacting && start_ + size_ + bytes > buffer_size_) {
    std::memmove(base_, base_ + start_, size_);
    start_ = 0;
  }
  // In the mirrored mode, `start_ < buffer_size_` and
  // `size_ + bytes <= capacity_ <= buffer_size_`, so the new bytes are always
  // inside the two copies of the memory.
  std::byte* const new_bytes = base_ + start_ + size_;
  size_ += bytes;
  return new_bytes;
}

absl::Status RingBufferStorage::Resize(const size_t bytes) {
  if (bytes > capacity_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Can't resize a RingBufferStorage of capacity ", capacity_,
                     " to ", bytes, " bytes."));
  }
  if (bytes > size_) {
    // `bytes <= capacity_`, so this never drops front bytes.
    return ExtendBack(bytes - size_).status();
  }
  size_ = bytes;
  if (mode_ == Mode::kCompacting && size_ == 0) {
    start_ = 0;
  }
  return absl::OkStatus();
}

absl::Status RingBufferStorage::PopFront(const size_t bytes) {
  if (bytes > size_) {
    return absl::OutOfRangeError(
        absl::StrCat("Can't pop ", bytes, " bytes from a RingBufferStorage of ",
                     size_, " bytes."));
  }
  DropFront(bytes);
  return absl::OkStatus();
}

void RingBufferStorage::DropFront(const size_t bytes) {
  start_ += bytes;
  size_ -= bytes;
  if (mode_ == Mode::kMirrored) {
    // Wraps around to the first copy of the memory. `buffer_size_ > 0` since
    // the capacity is positive.
    while (start_ >= buffer_size_) {
      start_ -= buffer_size_;
    }
  } else if (size_ == 0) {
    start_ = 0;
  }
}

void RingBufferStorage::Clear() {
  start_ = 0;
  size_ = 0;
}

void RingBufferStorage::Release() {
  if (base_ == nullptr) {
    return;
  }
  if (mode_ == Mode::kCompacting) {
    ::operator delete(base_, kCompactingAlignVal);
  } else {
#ifdef LITERT_TENSOR_RING_BUFFER_MIRRORED_SUPPORTED
    munmap(base_, allocation_size_);
#endif
  }
  base_ = nullptr;
  start_ = 0;
  size_ = 0;
}

}  // namespace litert::tensor::examples::internal
