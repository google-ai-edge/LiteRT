// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/samsung/dispatch/litert_weight_binary_manager.h"

#include <unistd.h>

#include <cstring>

#include "litert/c/internal/litert_logging.h"

namespace litert::samsung {

WeightBinaryManager& WeightBinaryManager::GetInstance(EnnManager* enn_manager) {
  static WeightBinaryManager instance(enn_manager);
  return instance;
}

WeightBinaryManager::WeightBinaryManager(EnnManager* enn_manager)
    : enn_manager_(enn_manager) {}

WeightBinaryManager::~WeightBinaryManager() {
  ClearAll();
  if (!enn_manager_->_enn_deinitialized_) {
    enn_manager_->_enn_deinitialized_ = true;
    enn_manager_->Api().EnnDeinitialize();
  }
}

// Returns cached buffer if exists, otherwise creates new (from file-opened fd)
litert::Expected<EnnBufferPtr> litert::samsung::WeightBinaryManager::Acquire(
    const std::string& signature, int fd, int64_t offset, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check cache first
  auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
    return e.signature == signature;
  });

  if (it != entries_.end()) {
    it->ref_count++;
    LITERT_LOG(LITERT_VERBOSE, "WeightBinaryManager: cache hit - %s",
               signature.c_str());
    return it->buffer;
  }

  // Cache miss - create new buffer
  EnnBufferPtr buffer;
  if (enn_manager_->Api().EnnCreateBufferCache(size, &buffer) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create weight buffer");
  }

  // Load data from file
  ssize_t bytes_read = pread(fd, buffer->va, size, offset);
  if (bytes_read != static_cast<ssize_t>(size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to read weight data");
  }

  entries_.push_back({signature, std::move(buffer), 1});
  LITERT_LOG(LITERT_VERBOSE, "WeightBinaryManager: created - %s",
             signature.c_str());

  return entries_.back().buffer;
}

// Decrements ref count
litert::Expected<void> WeightBinaryManager::Release(
    const std::string& signature) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
    return e.signature == signature;
  });

  if (it == entries_.end()) {
    return litert::Error(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  it->ref_count--;
  int new_count = it->ref_count;

  if (new_count <= 0) {
    if (it->buffer) {
      enn_manager_->Api().EnnReleaseBuffer(it->buffer);
    }
    entries_.erase(it);
  }

  return {};
}

// Clears all cached weights
void WeightBinaryManager::ClearAll() {
  std::lock_guard<std::mutex> lock(mutex_);

  // enn_manager_ may be dangling if EnnManager was destroyed first
  // (static destruction order across TUs is unspecified).
  // In that case, ENN runtime is already shutting down -- no-op is safe.
  if (!enn_manager_) return;

  for (auto& e : entries_) {
    if (e.buffer) {
      enn_manager_->Api().EnnReleaseBuffer(e.buffer);
    }
  }
  entries_.clear();
}

litert::Expected<EnnBufferPtr> WeightBinaryManager::Acquire(
    const std::string& signature, const void* address, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check cache first
  auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
    return e.signature == signature;
  });

  if (it != entries_.end()) {
    it->ref_count++;
    LITERT_LOG(LITERT_VERBOSE, "WeightBinaryManager: cache hit - %s",
               signature.c_str());
    return it->buffer;
  }

  // Cache miss - create new buffer
  EnnBufferPtr buffer;
  if (enn_manager_->Api().EnnCreateBufferCache(size, &buffer) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create weight buffer");
  }

  // Copy data from memory address into buffer
  if (address == nullptr) {
    enn_manager_->Api().EnnReleaseBuffer(buffer);
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Null address provided for weight acquisition");
  }
  std::memcpy(buffer->va, address, size);

  entries_.push_back({signature, std::move(buffer), 1});
  LITERT_LOG(LITERT_VERBOSE, "WeightBinaryManager: created from memory - %s",
             signature.c_str());

  return entries_.back().buffer;
}

}  // namespace litert::samsung
