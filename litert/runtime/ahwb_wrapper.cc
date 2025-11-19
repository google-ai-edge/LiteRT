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

#include "litert/runtime/ahwb_wrapper.h"

#ifdef __ANDROID__
#include <dlfcn.h>
#endif  // __ANDROID__

namespace litert::internal {

AndroidHardwareBufferWrapper::AndroidHardwareBufferWrapper() {
#ifdef __ANDROID__
  dlopen_handle_ = dlopen("libnativewindow.so", RTLD_NOW);
  if (dlopen_handle_ == nullptr) {
    supported_ = false;
    return;
  }
  allocate_ = reinterpret_cast<decltype(allocate_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_allocate"));
  acquire_ = reinterpret_cast<decltype(acquire_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_acquire"));
  release_ = reinterpret_cast<decltype(release_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_release"));
  describe_ = reinterpret_cast<decltype(describe_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_describe"));
  is_supported_ = reinterpret_cast<decltype(is_supported_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_isSupported"));
  lock_ = reinterpret_cast<decltype(lock_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_lock"));
  unlock_ = reinterpret_cast<decltype(unlock_)>(
      dlsym(dlopen_handle_, "AHardwareBuffer_unlock"));
  supported_ =
      (allocate_ != nullptr && acquire_ != nullptr && release_ != nullptr &&
       describe_ != nullptr && is_supported_ != nullptr && lock_ != nullptr &&
       unlock_ != nullptr);
#else
  dlopen_handle_ = nullptr;
  allocate_ = nullptr;
  acquire_ = nullptr;
  release_ = nullptr;
  describe_ = nullptr;
  is_supported_ = nullptr;
  supported_ = false;
#endif
}

}  // namespace litert::internal
