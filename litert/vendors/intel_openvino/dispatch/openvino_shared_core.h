// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_

#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

// Bundles the OpenVINO core and all hardware-context-derived state whose
// lifetimes must be tied together. Destroying this object releases the
// underlying Level Zero / NPU resources.
class OpenVINOCore {
 public:
  OpenVINOCore() : core_(std::make_shared<ov::Core>()) {}
  OpenVINOCore(const OpenVINOCore&) = delete;
  OpenVINOCore(OpenVINOCore&&) = delete;
  OpenVINOCore& operator=(const OpenVINOCore&) = delete;
  OpenVINOCore& operator=(OpenVINOCore&&) = delete;

  std::shared_ptr<ov::Core> core() const { return core_; }

  void SetDevice(const std::string& device) {
    absl::MutexLock lock(mutex_);
    device_ = device;
    remote_context_.reset();
  }

  std::string GetDevice() {
    absl::MutexLock lock(mutex_);
    return device_;
  }

  ov::RemoteContext GetRemoteContext() {
    absl::MutexLock lock(mutex_);
    if (!remote_context_.has_value()) {
      remote_context_ = core_->get_default_context(device_);
    }
    return *remote_context_;
  }

  const std::vector<std::string>& GetAvailableDevices();

 private:
  absl::Mutex mutex_;
  std::shared_ptr<ov::Core> core_;  // never null after construction
  std::string device_ ABSL_GUARDED_BY(mutex_) = "NPU";
  std::optional<ov::RemoteContext> remote_context_ ABSL_GUARDED_BY(mutex_);
  std::once_flag available_devices_once_;
  std::vector<std::string> available_devices_;
};

// Process-wide provider. Hands out a *shared* OpenVINOCore, created lazily and
// destroyed automatically once the last handle is dropped. No manual counting.
class OpenVINOSharedCore {
 public:
  using Handle = std::shared_ptr<OpenVINOCore>;

  OpenVINOSharedCore(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore(OpenVINOSharedCore&&) = delete;
  OpenVINOSharedCore& operator=(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore& operator=(OpenVINOSharedCore&&) = delete;

  static OpenVINOSharedCore* GetInstance();

  // Returns a strong handle to the live core, creating a fresh one if none is
  // currently alive. Callers keep the handle for as long as they use the core.
  Handle Acquire() {
    absl::MutexLock lock(mutex_);
    Handle core = weak_core_.lock();
    if (!core) {
      core = std::make_shared<OpenVINOCore>();
      weak_core_ = core;
    }
    return core;
  }

 private:
  OpenVINOSharedCore() = default;
  absl::Mutex mutex_;
  std::weak_ptr<OpenVINOCore> weak_core_ ABSL_GUARDED_BY(mutex_);
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
