// Copyright 2026 The LiteRT Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef LITERT_ARM_VULKAN_ML_EXTERNAL_BUFFER_H_
#define LITERT_ARM_VULKAN_ML_EXTERNAL_BUFFER_H_
#include <memory>
#include <string>
#include "vulkan/vulkan.h"
namespace litert::arm_vulkan_ml {
// Platform adapter also permits AHB lifecycle tests on non-Android hosts.
class HardwareBufferApi {
 public:
  virtual ~HardwareBufferApi() = default;
  virtual bool Describe(void*, uint64_t& bytes, uint64_t& id) = 0;
  virtual void Acquire(void*) = 0;
  virtual void Release(void*) = 0;
  virtual bool Properties(VkDevice, PFN_vkGetDeviceProcAddr, void*,
                          VkDeviceSize&, uint32_t&) = 0;
};
std::shared_ptr<HardwareBufferApi> AndroidHardwareBufferApi();
struct ExternalBuffer {
  enum class Kind { kDmaBuf, kAhb };
  Kind kind;
  int fd = -1;
  void* ahb = nullptr;
  size_t size = 0, offset = 0;
  uint64_t identity_a = 0, identity_b = 0;
  std::shared_ptr<HardwareBufferApi> api;
  ~ExternalBuffer();
  static std::shared_ptr<ExternalBuffer> DmaBuf(int fd, size_t size,
      size_t offset, std::string& error);
  static std::shared_ptr<ExternalBuffer> Ahb(void*, size_t size, size_t offset,
      std::shared_ptr<HardwareBufferApi>, std::string& error);
  VkExternalMemoryHandleTypeFlagBits HandleType() const;
  uint32_t ForeignQueue() const;
  bool Overlaps(const ExternalBuffer&, size_t bytes, size_t other_bytes) const;
};
}
#endif
