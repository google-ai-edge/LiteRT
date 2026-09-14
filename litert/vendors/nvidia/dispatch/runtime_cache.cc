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

#include "litert/vendors/nvidia/dispatch/runtime_cache.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "NvInferRuntime.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::nvidia {
namespace {

Error FileError(const char* operation) {
  const int error = errno;
  return Error(kLiteRtStatusErrorFileIO,
               std::string(operation) + ": " + std::strerror(error));
}

}  // namespace

Expected<void> PersistValidatedRuntimeCache(nvinfer1::IRuntimeConfig& config,
                                          const void* data, size_t size,
                                          const std::string& path) {
  if (data == nullptr || size == 0 || path.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Empty TensorRT runtime cache or destination");
  }
  std::unique_ptr<nvinfer1::IRuntimeCache> probe(config.createRuntimeCache());
  if (!probe) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create TensorRT runtime cache validation probe");
  }
  if (!probe->deserialize(data, size)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT rejected serialized runtime cache; "
                 "keeping old file");
  }
  // SDK 1.5 can accept a cache, then serialize it into bytes that it rejects on
  // the next reload. Persist the validated candidate, never probe->serialize().
  // Validation does not replace the live cache or freeze later shape additions.
  probe.reset();

  // Write beside the destination so rename is atomic. A regenerable cache does
  // not need crash-durable fsync. Never truncate a usable file on failure.
  std::string temporary_path = path + ".tmp.XXXXXX";
  const int fd = mkstemp(temporary_path.data());
  if (fd < 0) {
    return FileError("Creating temporary TensorRT runtime cache");
  }
  std::FILE* file = fdopen(fd, "wb");
  if (file == nullptr) {
    const Error error = FileError("Opening temporary TensorRT runtime cache");
    close(fd);
    unlink(temporary_path.c_str());
    return error;
  }
  if (std::fwrite(data, 1, size, file) != size) {
    const Error error = FileError("Writing temporary TensorRT runtime cache");
    std::fclose(file);
    unlink(temporary_path.c_str());
    return error;
  }
  if (std::fclose(file) != 0) {
    const Error error = FileError("Closing temporary TensorRT runtime cache");
    unlink(temporary_path.c_str());
    return error;
  }
  if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
    const Error error = FileError("Replacing TensorRT runtime cache");
    unlink(temporary_path.c_str());
    return error;
  }
  return {};
}

}  // namespace litert::nvidia
