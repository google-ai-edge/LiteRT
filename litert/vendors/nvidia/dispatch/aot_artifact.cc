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

#include "litert/vendors/nvidia/dispatch/aot_artifact.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::nvidia {
namespace {

struct ValidatedAotArtifact {
  TensorRtAotFileIdentity file_identity;
  uint64_t size = 0;
  TensorRtArtifactFingerprint fingerprint;
};

TensorRtAotFileIdentity FileIdentity(const struct stat& stat_buffer) {
  return {static_cast<uint64_t>(stat_buffer.st_dev),
          static_cast<uint64_t>(stat_buffer.st_ino),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_nsec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_nsec)};
}

bool SameArtifact(const ValidatedAotArtifact& lhs,
                  const ValidatedAotArtifact& rhs) {
  return lhs.file_identity == rhs.file_identity && lhs.size == rhs.size &&
         lhs.fingerprint == rhs.fingerprint;
}

bool ForceContentValidation() {
  const char* value =
      std::getenv("LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

Expected<TensorRtArtifactFingerprint> FingerprintOpenArtifact(
    int fd, const void* mapping, size_t size, uint32_t locator_version) {
  if (locator_version == 1) {
    return FingerprintTensorRtArtifact(mapping, size);
  }
  TensorRtAotFingerprintBuilder builder;
  std::vector<uint8_t> buffer(kTensorRtAotFingerprintChunkBytes);
  size_t offset = 0;
  while (offset < size) {
    const size_t requested = std::min(buffer.size(), size - offset);
    size_t received = 0;
    while (received < requested) {
      const ssize_t result =
          pread(fd, buffer.data() + received, requested - received,
                static_cast<off_t>(offset + received));
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        return Error(kLiteRtStatusErrorFileIO,
                     "Failed to stream TensorRT AOT artifact fingerprint: " +
                         std::string(result == 0 ? "unexpected EOF"
                                                 : std::strerror(errno)));
      }
      received += static_cast<size_t>(result);
    }
    builder.Add(buffer.data(), received);
    posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(received),
                  POSIX_FADV_DONTNEED);
    offset += received;
  }
  return builder.Finish();
}

}  // namespace

const char* AotArtifactValidationName(AotArtifactValidation validation) {
  switch (validation) {
    case AotArtifactValidation::kTrustedFileIdentity:
      return "trusted_file_identity";
    case AotArtifactValidation::kComputedFingerprint:
      return "computed_fingerprint";
    case AotArtifactValidation::kProcessCache:
      return "process_cache";
  }
  return "unknown";
}

Expected<std::unique_ptr<MappedAotArtifact>> MappedAotArtifact::Open(
    const TensorRtAotLocator& locator) {
  if (locator.artifact_size == 0 ||
      locator.artifact_size > std::numeric_limits<size_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT artifact size is invalid");
  }
  const int fd = open(locator.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to open TensorRT AOT artifact " + locator.path + ": " +
                     std::strerror(errno));
  }
  struct stat stat_buffer{};
  if (fstat(fd, &stat_buffer) != 0) {
    const int saved_errno = errno;
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to inspect TensorRT AOT artifact " + locator.path +
                     ": " + std::strerror(saved_errno));
  }
  if (!S_ISREG(stat_buffer.st_mode) || stat_buffer.st_size < 0 ||
      static_cast<uint64_t>(stat_buffer.st_size) != locator.artifact_size) {
    close(fd);
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT artifact size or file type does not match "
                 "its locator");
  }
  const size_t size = static_cast<size_t>(locator.artifact_size);
  void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int mmap_errno = errno;
  if (data == MAP_FAILED) {
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to map TensorRT AOT artifact " + locator.path + ": " +
                     std::strerror(mmap_errno));
  }

  const auto actual_identity = FileIdentity(stat_buffer);
  const bool force_content_validation = ForceContentValidation();
  const bool artifact_is_read_only =
      (stat_buffer.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0;
  if (!force_content_validation && artifact_is_read_only &&
      locator.file_identity.has_value() &&
      *locator.file_identity == actual_identity) {
    close(fd);
    return std::unique_ptr<MappedAotArtifact>(new MappedAotArtifact(
        data, size, AotArtifactValidation::kTrustedFileIdentity));
  }

  static std::mutex validation_mutex;
  static std::unordered_map<std::string, ValidatedAotArtifact>
      validated_artifacts;
  std::lock_guard<std::mutex> lock(validation_mutex);
  const ValidatedAotArtifact actual_artifact{
      actual_identity, static_cast<uint64_t>(size), locator.fingerprint};
  if (!force_content_validation) {
    const auto cached = validated_artifacts.find(locator.path);
    if (cached != validated_artifacts.end() &&
        SameArtifact(cached->second, actual_artifact)) {
      close(fd);
      return std::unique_ptr<MappedAotArtifact>(new MappedAotArtifact(
          data, size, AotArtifactValidation::kProcessCache));
    }
  }
  auto actual_fingerprint =
      FingerprintOpenArtifact(fd, data, size, locator.version);
  if (!actual_fingerprint) {
    munmap(data, size);
    close(fd);
    return actual_fingerprint.Error();
  }
  struct stat validated_stat{};
  if (fstat(fd, &validated_stat) != 0) {
    const int saved_errno = errno;
    munmap(data, size);
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to inspect TensorRT AOT artifact after validation: " +
                     std::string(std::strerror(saved_errno)));
  }
  if (!(FileIdentity(validated_stat) == actual_identity)) {
    munmap(data, size);
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "TensorRT AOT artifact changed during validation");
  }
  if (!(*actual_fingerprint == locator.fingerprint)) {
    munmap(data, size);
    close(fd);
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT artifact fingerprint does not match its "
                 "locator");
  }
  close(fd);
  validated_artifacts[locator.path] = actual_artifact;
  return std::unique_ptr<MappedAotArtifact>(new MappedAotArtifact(
      data, size, AotArtifactValidation::kComputedFingerprint));
}

MappedAotArtifact::~MappedAotArtifact() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
}

}  // namespace litert::nvidia
