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

#ifndef LITERT_RUNTIME_ANDROID_HARDWARE_BUFFER_WRAPPER_H_
#define LITERT_RUNTIME_ANDROID_HARDWARE_BUFFER_WRAPPER_H_

#include <stdint.h>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#else
extern "C" {
typedef struct AHardwareBuffer AHardwareBuffer;

// struct is a copy of the Android NDK AHardwareBuffer_Desc struct in the link
// below
// https://developer.android.com/ndk/reference/struct/a-hardware-buffer-desc
typedef struct AHardwareBuffer_Desc AHardwareBuffer_Desc;
struct AHardwareBuffer_Desc {
  uint32_t width;
  uint32_t height;
  uint32_t layers;
  uint32_t format;
  uint64_t usage;
  uint32_t stride;
  uint32_t rfu0;
  uint64_t rfu1;
};

enum AHardwareBuffer_Format {
  AHARDWAREBUFFER_FORMAT_BLOB = 0x21,
};

enum AHardwareBuffer_UsageFlags {
  AHARDWAREBUFFER_USAGE_CPU_READ_RARELY = 2UL,
  AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY = 2UL << 4,
  AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER = 1UL << 24
};

}  // extern "C"
#endif  // __ANDROID__

namespace litert::internal {

// This header file and singleton class encapsulates the following Android NDK
// features
//   - header <android/hardware_buffer.h>
//   - opaque struct type AHardwareBuffer
//   - struct type AHardwareBuffer_Desc
//   - function AHardwareBuffer_isSupported
//   - function AHardwareBuffer_allocate
//   - function AHardwareBuffer_acquire
//   - function AHardwareBuffer_release
//   - function AHardwareBuffer_describe
//   - function AHardwareBuffer_lock
//   - function AHardwareBuffer_unlock
//   - library libnativewindow.so (for the above features)
//
// For documentation on these features, see
// <https://developer.android.com/ndk/reference/group/a-hardware-buffer>:
//
// Unlike using the native NDK functionality directly, this class only has a
// run-time dependency on API level 26, not a build-time dependency.  So it can
// be used even when building with NDK min SDK level < 26, as long as you are
// very careful to check that Supported() returns true before calling any other
// methods.
class AndroidHardwareBufferWrapper {
 public:
  static AndroidHardwareBufferWrapper& Instance() {
    static AndroidHardwareBufferWrapper instance;
    return instance;
  }

  // Returns true if the functionality in this class is supported.
  bool Supported() { return supported_; }

  // Like AHardwareBuffer_isSupported.
  // Caller must check that Supported() returns true before calling this
  // function.
  int IsSupported(const AHardwareBuffer_Desc* description) {
    return is_supported_(description);
  }

  // Like AHardwareBuffer_allocate.
  // Caller must check that Supported() returns true before calling this
  // function.
  int Allocate(const AHardwareBuffer_Desc* description,
               AHardwareBuffer** buffer) {
    return allocate_(description, buffer);
  }

  // Like AHardwareBuffer_acquire.
  // Caller must check that Supported() returns true before calling this
  // function.
  void Acquire(AHardwareBuffer* buffer) { return acquire_(buffer); }

  // Like AHardwareBuffer_release.
  // Caller must check that Supported() returns true before calling this
  // function.
  void Release(AHardwareBuffer* buffer) { return release_(buffer); }

  // Like AHardwareBuffer_describe.
  // Caller must check that Supported() returns true before calling this
  // function.
  void Describe(AHardwareBuffer* buffer, AHardwareBuffer_Desc* desc) {
    return describe_(buffer, desc);
  }

  // Like AHardwareBuffer_lock.
  // Caller must check that Supported() returns true before calling this
  // function.
  int Lock(AHardwareBuffer* buffer, uint64_t usage, int32_t fence,
           const void* rect, void** vaddr) {
    return lock_(buffer, usage, fence, rect, vaddr);
  }

  // Like AHardwareBuffer_unlock.
  // Caller must check that Supported() returns true before calling this
  // function.
  int Unlock(AHardwareBuffer* buffer, int32_t* fence) {
    return unlock_(buffer, fence);
  }

 private:
  void* dlopen_handle_;
  int (*is_supported_)(const AHardwareBuffer_Desc* desc);
  int (*allocate_)(const AHardwareBuffer_Desc* desc, AHardwareBuffer** buffer);
  void (*acquire_)(AHardwareBuffer* buffer);
  void (*release_)(AHardwareBuffer* buffer);
  void (*describe_)(AHardwareBuffer* buffer, AHardwareBuffer_Desc* desc);
  int (*lock_)(AHardwareBuffer* buffer, uint64_t usage, int32_t fence,
               const void* rect, void** vaddr);
  int (*unlock_)(AHardwareBuffer* buffer, int32_t* fence);
  bool supported_;

  AndroidHardwareBufferWrapper();
  AndroidHardwareBufferWrapper(const AndroidHardwareBufferWrapper&) = delete;
  // Note that we deliberately do not call dlclose() in the destructor; doing
  // so would complicate the code and would unnecessarily introduce additional
  // failure scenarios. The object is a singleton and so is only destroyed when
  // the process is about to exit, and the OS will automatically reclaim the
  // resources on process exit anyway, so calling dlclose would only slow down
  // process exit.
  ~AndroidHardwareBufferWrapper() = default;
};

}  // namespace litert::internal

#endif  // LITERT_RUNTIME_ANDROID_HARDWARE_BUFFER_WRAPPER_H_
